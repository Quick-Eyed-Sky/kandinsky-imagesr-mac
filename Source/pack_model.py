"""Bake the released timestep projections once and pack weights for the local MLX engine."""
import json
from pathlib import Path
import shutil
import sys
import mlx.core as mx
import mlx.nn as nn


def pack(source, out):
    transformer = source / "transformer"
    out.mkdir(parents=True,exist_ok=True)
    cfg = json.loads((transformer / "config.json").read_text())
    shards = sorted(transformer.glob("*.safetensors"))
    index = json.loads((transformer / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    times = nn.silu(mx.load(str(transformer / index["time_embed_table"]))["time_embed_table"])
    mx.eval(times)
    for n,file in enumerate(shards):
        weights = mx.load(str(file))
        packed = {}
        for key, value in weights.items():
            # All released timesteps are retained. Drop the now-constant projection matrices.
            if key.endswith("modulation.out_layer.weight"):
                prefix = key.removesuffix(".weight")
                bias_key = prefix + ".bias"
                bias = weights.get(bias_key)
                if bias is None:
                    bias = mx.load(str(transformer / index[bias_key]))[bias_key]
                packed[prefix+".table"] = mx.addmm(bias,times,value.T)
            elif key.endswith("modulation.out_layer.bias") or key == "time_embed_table":
                continue
            else:
                packed[key] = value.astype(mx.float16) if value.dtype == mx.bfloat16 else value
        mx.save_safetensors(str(out / f"blocks-{n}.safetensors"),packed)
    (out/"config.json").write_text(json.dumps(cfg,indent=2))
    shutil.copytree(source/"vae",out/"vae",dirs_exist_ok=True)
    for name in ("README.md","SOURCE.json","model_index.json"):
        shutil.copy2(source/name,out/name)


if __name__ == "__main__":
    pack(Path(sys.argv[1]),Path(sys.argv[2]))
