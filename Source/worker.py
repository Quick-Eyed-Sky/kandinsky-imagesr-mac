"""Local batch protocol. One JSON request per launch, newline-delimited progress on stdout."""
from functools import lru_cache
import io
import json
import os
import signal
from pathlib import Path
import sys
import tempfile
import time
import traceback

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
from metal_model import MetalModel

register_heif_opener()
TIMESTEPS = {2:(.07,.086,.118,.134,.15,.18,.21,.27,.30),
             4:(.17,.1885,.207,.2255,.244,.278,.312,.346,.38)}
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


def emit(event, **data):
    print(json.dumps(dict(event=event, **data)), flush=True)


def pixels_and_alpha(path):
    with Image.open(path) as source:
        im = ImageOps.exif_transpose(source)
        alpha = im.convert("RGBA").getchannel("A") if im.has_transparency_data else None
        profile = im.info.get("icc_profile")
        if profile:
            try:
                color = im if im.mode in ("RGB","CMYK","L","LAB") else im.convert("RGB")
                im = ImageCms.profileToProfile(color,ImageCms.ImageCmsProfile(io.BytesIO(profile)),SRGB,
                                               renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,outputMode="RGB")
            except (OSError, ValueError, ImageCms.PyCMSError):
                pass
        return np.asarray(im.convert("RGB")), alpha


@lru_cache(maxsize=2)
def cubic_table(factor):
    # PyTorch's align_corners=False bicubic kernel (a = -0.75).
    pos = (np.arange(1024,dtype=np.float32)+0.5)/factor-0.5
    index = np.floor(pos).astype(np.int32)[:,None] + np.arange(-1,3)
    d = np.abs(pos[:,None]-index)
    coeff = np.where(d<=1, (1.25*d-2.25)*d*d+1, ((-0.75*d+3.75)*d-6)*d+3)
    return index.clip(0,1024//factor-1), coeff.astype(np.float32)


def bicubic(tile, factor):
    idx, coeff = cubic_table(factor)
    tmp = sum(tile[:,idx[:,i]] * coeff[None,:,i,None] for i in range(4))
    out = sum(tmp[idx[:,i]] * coeff[:,i,None,None] for i in range(4))
    return np.clip(out/255,0,1)


def starts(length, size):
    last = max(0,length-size)
    return [*range(0,last,3*size//4),last]


@lru_cache(maxsize=4)
def window(first,last):
    a = (0.5-0.5*np.cos(2*np.pi*np.arange(1024)/1024)).astype(np.float32)
    if first:
        a[:512] = 1
    if last:
        a[512:] = 1
    return a


def upscale(model, path, factor, level, progress):
    image, alpha = pixels_and_alpha(path)
    h,w = image.shape[:2]
    tile = 1024//factor
    rows, cols = starts(h,tile),starts(w,tile)
    image = np.pad(image,((0,max(0,tile-h)),(0,max(0,tile-w)),(0,0)),mode="edge")
    # Only accumulate overlapping scanlines in float32, even on very tall originals.
    acc = np.zeros((1024, image.shape[1]*factor,3),np.float32)
    weight = np.zeros((*acc.shape[:2],1),np.float32)
    output = np.empty((h*factor,w*factor,3),np.uint8)
    count = len(rows)*len(cols)
    for r,top in enumerate(rows):
        wy = window(r==0,r==len(rows)-1)
        for c,left in enumerate(cols):
            progress(r*len(cols)+c,count)
            pixels = bicubic(image[top:top+tile,left:left+tile],factor)
            predicted = np.asarray(model.tile(pixels,TIMESTEPS[factor][level]))
            if not np.isfinite(predicted).all():
                raise ValueError("The GPU produced invalid values.")
            mask = (wy[:,None]*window(c==0,c==len(cols)-1)[None,:])[:,:,None]
            region = slice(left*factor,left*factor+1024)
            acc[:,region] += predicted * mask
            weight[:,region] += mask
        start = top*factor
        end = rows[r+1]*factor if r+1<len(rows) else h*factor
        n = end-start
        output[start:end] = (acc[:n,:w*factor] / np.maximum(weight[:n,:w*factor],1e-6)*255).clip(0,255)
        if r+1<len(rows):
            acc[:-n], weight[:-n] = acc[n:].copy(),weight[n:].copy()
            acc[-n:] = 0
            weight[-n:] = 0
    result = Image.fromarray(output)
    if alpha is not None:
        result.putalpha(alpha.resize(result.size,Image.Resampling.LANCZOS))
    return result


def save_unique(image,source,folder,factor,fmt):
    suffix = ".jpg" if fmt=="JPEG" else ".png"
    base = f"{source.stem}_KSR_x{factor}"
    # Publish an encoded file atomically, including when two exports share a name.
    with tempfile.NamedTemporaryFile(dir=folder, prefix=".ksr-", suffix=".tmp") as tmp:
        if fmt=="JPEG" and image.mode=="RGBA":
            bg = Image.new("RGB", image.size,"white")
            bg.paste(image,mask=image.getchannel("A"))
            image = bg
        image.save(tmp, format=fmt, icc_profile=SRGB.tobytes(), quality=95, subsampling=0, compress_level=2)
        tmp.flush()
        n = 1
        while True:
            dst = folder / (base + ("" if n==1 else f"-{n}") + suffix)
            try:
                os.link(tmp.name,dst)
                return dst
            except FileExistsError:
                n += 1
            except OSError as exc:
                # exFAT / network volumes may not support hard links.
                import errno, shutil
                if exc.errno not in (errno.EOPNOTSUPP, errno.EPERM, errno.ENOSYS, errno.EXDEV):
                    raise
                try:
                    f = dst.open("xb")
                except FileExistsError:
                    n += 1
                    continue
                try:
                    with f, open(tmp.name,"rb") as saved:
                        shutil.copyfileobj(saved,f)
                except BaseException:
                    dst.unlink(missing_ok=True)
                    raise
                return dst


def main():
    request = json.load(sys.stdin)
    model = MetalModel(Path(sys.argv[1]))
    for index,name in enumerate(request["files"]):
        path = Path(name)
        try:
            t = time.perf_counter()
            factor = request["factor"]
            result = upscale(model,path,factor,request["level"],
                             lambda tile,total: emit("tile",index=index,tile=tile,total=total))
            dst = save_unique(result,path,Path(request["folder"]) if request["folder"] else path.parent,
                              factor,request["format"])
            emit("done",index=index,path=str(dst),seconds=time.perf_counter()-t)
        except UnidentifiedImageError:
            emit("failed",index=index,reason="The file does not contain a readable image.")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit("failed",index=index,reason=str(exc))


def stop(signum, frame):
    # Unwind open temp files when the user stops the current batch.
    raise SystemExit(128 + signum)


if __name__=="__main__":
    signal.signal(signal.SIGTERM, stop)
    main()
