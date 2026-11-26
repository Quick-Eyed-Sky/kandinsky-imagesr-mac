"""Apple MLX inference for the released, one-step Kandinsky ImageSR 2.0 weights."""
import json
import math

import mlx.core as mx
import mlx.nn as nn

HALF = mx.float16
FULL = mx.float32


def linear(x, weights, name):
    matrix = weights[name + ".weight"]
    bias = weights.get(name + ".bias")
    return x @ matrix.T if bias is None else mx.addmm(bias, x, matrix.T)


@mx.compile
def modulate(x, shift, scale):
    normalized = mx.fast.layer_norm(x.astype(FULL), None, None, 1e-5)
    return (normalized * (1 + scale) + shift).astype(x.dtype)


@mx.compile
def gated(x, gate, delta):
    return (x.astype(FULL) + gate * delta.astype(FULL)).astype(x.dtype)


@mx.compile
def rotate(x, weight, cosine, sine):
    # Kandinsky normalizes Q/K in FP32, rounds, then rotates adjacent pairs.
    x = mx.fast.rms_norm(x.astype(FULL), weight.astype(FULL), 1.1920928955078125e-7).astype(x.dtype)
    pairs = x.astype(FULL).reshape(*x.shape[:-1], -1, 2)
    first, second = pairs[..., 0], pairs[..., 1]
    result = mx.stack((first*cosine - second*sine, first*sine + second*cosine), axis=-1)
    return result.reshape(x.shape).astype(HALF).transpose(0, 2, 1, 3)


class Transformer:
    def __init__(self, root):
        config = json.loads((root / "config.json").read_text())
        self.times = config["baked_timesteps"]
        self.layers = config["num_visual_blocks"]
        dims = config["axes_dims"]
        self.head = sum(dims)
        self.heads = config["model_dim"] // self.head
        weights = {}
        for file in sorted(root.glob("blocks-*.safetensors")):
            weights.update(mx.load(str(file)))
        self.weights = weights
        self.mods = [mx.split(weights[f"visual_transformer_blocks.{i}.visual_modulation.out_layer.table"], 9, axis=-1)
                     for i in range(self.layers)]
        self.final_mod = mx.split(weights["out_layer.modulation.out_layer.table"], 2, axis=-1)

        y, x = mx.meshgrid(mx.arange(64), mx.arange(64), indexing="ij")
        angles = []
        for pos, dim in zip((mx.zeros_like(y), y, x), dims):
            frequency = mx.exp(-math.log(10000) * mx.arange(dim // 2) / (dim // 2))
            angles.append(pos.reshape(-1, 1) * frequency)
        angle = mx.concatenate(angles, axis=-1).reshape(1, 4096, 1, -1)
        self.cos, self.sin = mx.cos(angle), mx.sin(angle)
        mx.eval(self.weights, self.final_mod, self.cos, self.sin)

    def __call__(self, latent, t):
        idx = self.times.index(t)
        w = self.weights
        x = latent.astype(HALF).reshape(1,64,2,64,2,16).transpose(0,1,3,2,4,5).reshape(1,4096,64)
        x = linear(x, w, "visual_embeddings.in_layer")
        for layer, mods in enumerate(self.mods):
            prefix = f"visual_transformer_blocks.{layer}."
            attention = prefix + "self_attention."
            shift, scale, gate, _, _, _, ff_shift, ff_scale, ff_gate = (m[idx] for m in mods)
            normalized = modulate(x, shift, scale)
            def heads(name):
                return linear(normalized, w, attention + name).reshape(1,4096,self.heads,self.head)
            q = rotate(heads("to_query"), w[attention+"query_norm.weight"], self.cos, self.sin)
            k = rotate(heads("to_key"), w[attention+"key_norm.weight"], self.cos, self.sin)
            v = heads("to_value").transpose(0,2,1,3)
            a = mx.fast.scaled_dot_product_attention(q,k,v,scale=self.head**-0.5)
            a = a.transpose(0,2,1,3).reshape(x.shape)
            x = gated(x, gate, linear(a, w, attention + "out_layer"))
            x = (x.astype(FULL) + w[prefix + "xattn_bias_table"][idx]).astype(HALF)
            ff = linear(modulate(x, ff_shift, ff_scale), w, prefix + "feed_forward.in_layer")
            x = gated(x, ff_gate, linear(nn.gelu(ff), w, prefix + "feed_forward.out_layer"))
            # Bound graph depth and release intermediate activations between blocks.
            mx.async_eval(x)
        shift, scale = (p[idx] for p in self.final_mod)
        out = linear(modulate(x, shift, scale), w, "out_layer.out_layer")
        return out.reshape(1,64,64,16,2,2).transpose(0,1,4,2,5,3).reshape(1,128,128,16)


class Autoencoder:
    def __init__(self, root):
        self.weights = mx.load(str(root / "diffusion_pytorch_model.safetensors"))
        for key, weight in list(self.weights.items()):
            if weight.ndim == 4:
                self.weights[key] = mx.contiguous(weight.transpose(0,2,3,1))
        self.scaling = json.loads((root / "config.json").read_text())["scaling_factor"]
        mx.eval(self.weights)

    def conv(self, x, name, stride=1, pad=1):
        w = self.weights[name + ".weight"]
        return mx.conv2d(x, w, stride=stride, padding=pad) + self.weights[name + ".bias"]

    def norm(self, x, name):
        groups = x.reshape(x.shape[0], -1, 32, x.shape[-1]//32).transpose(0,2,1,3)
        norm = mx.fast.layer_norm(groups.reshape(x.shape[0],32,-1), None, None, 1e-6)
        norm = norm.reshape(groups.shape).transpose(0,2,1,3).reshape(x.shape)
        return (norm.astype(FULL) * self.weights[name+".weight"] + self.weights[name+".bias"]).astype(x.dtype)

    def resnet(self, x, name):
        h = self.conv(nn.silu(self.norm(x,name+".norm1")),name+".conv1")
        h = self.conv(nn.silu(self.norm(h,name+".norm2")),name+".conv2")
        if name+".conv_shortcut.weight" in self.weights:
            x = self.conv(x, name+".conv_shortcut", pad=0)
        return x+h

    def mid(self,x,side):
        base = side + ".mid_block."
        x = self.resnet(x, base+"resnets.0")
        key = base+"attentions.0."
        h = self.norm(x, key+"group_norm").reshape(1,-1,x.shape[-1])
        q,k,v = (linear(h,self.weights,key+"to_"+p)[:,None] for p in ("q","k","v"))
        # The FLUX VAE uses one 512-wide head. Chunk its queries to bound scratch memory.
        chunks = [mx.fast.scaled_dot_product_attention(q[:,:,i:i+256], k, v, scale=h.shape[-1]**-0.5)
                  for i in range(0,h.shape[1],256)]
        attn = mx.concatenate(chunks,axis=2)[:,0]
        x = x + linear(attn,self.weights,key+"to_out.0").reshape(x.shape)
        return self.resnet(x,base+"resnets.1")

    def encode(self, pixels):
        x = self.conv((pixels*2-1).astype(mx.bfloat16), "encoder.conv_in")
        for i in range(4):
            base = f"encoder.down_blocks.{i}."
            for j in range(2):
                x = self.resnet(x,base+f"resnets.{j}")
            if i < 3:
                x = mx.pad(x, [(0,0),(0,1),(0,1),(0,0)])
                x = self.conv(x,base+"downsamplers.0.conv",stride=2,pad=0)
            mx.async_eval(x)
        x = self.mid(x,"encoder")
        x = self.conv(nn.silu(self.norm(x,"encoder.conv_norm_out")),"encoder.conv_out")
        return x[...,:16].astype(FULL)*self.scaling

    def decode(self, latent):
        x = self.conv((latent/self.scaling).astype(mx.bfloat16),"decoder.conv_in")
        x = self.mid(x,"decoder")
        for i in range(4):
            base = f"decoder.up_blocks.{i}."
            for j in range(3):
                x = self.resnet(x,base+f"resnets.{j}")
            if i < 3:
                x = mx.repeat(mx.repeat(x,2,axis=1),2,axis=2)
                x = self.conv(x,base+"upsamplers.0.conv")
            mx.async_eval(x)
        x = self.conv(nn.silu(self.norm(x,"decoder.conv_norm_out")),"decoder.conv_out")
        return (mx.clip(x.astype(FULL),-1,1)+1)*0.5


class MetalModel:
    def __init__(self, root):
        self.vae = Autoencoder(root/"vae")
        self.transformer = Transformer(root)
        mx.set_cache_limit(2 * 1024**3)

    def tile(self, pixels, timestep):
        encoded = self.vae.encode(mx.array(pixels)[None])
        delta = self.transformer(encoded, timestep)
        return self.vae.decode(encoded - timestep * delta.astype(FULL))[0]
