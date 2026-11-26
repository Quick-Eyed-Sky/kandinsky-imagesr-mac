"""Generate the layered landscape/loupe artwork and Finder icon bitmaps."""
from pathlib import Path
import sys

from PIL import Image, ImageChops, ImageDraw, ImageFilter

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
art = Image.new("RGBA",(1024,1024))
shadow = Image.new("RGBA",art.size)
ImageDraw.Draw(shadow).rounded_rectangle((66,76,958,964),radius=199,fill=(13,12,20,120))
art.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(18)))
mask = Image.new("L",art.size)
ImageDraw.Draw(mask).rounded_rectangle((60,52,964,948),radius=197,fill=255)
field = Image.new("RGBA",art.size)
d = ImageDraw.Draw(field)
for y in range(1024):
    frac = y/1024
    d.line((0,y,1024,y),fill=(int(45-21*frac),int(48-20*frac),int(65-28*frac),255))
d.rounded_rectangle((66,58,958,942),radius=191,outline="#545769",width=4)
# An abstract photograph changes from large mosaic cells to sharp contours in the loupe.
land = Image.new("RGB",art.size,"#E7BF87")
g = ImageDraw.Draw(land)
g.ellipse((440,226,665,451),fill="#EC784F")
g.polygon([(0,700),(375,380),(515,539),(640,389),(1024,762),(1024,1024),(0,1024)],fill="#527E80")
g.polygon([(0,800),(465,565),(559,625),(782,498),(1024,630),(1024,1024),(0,1024)],fill="#284A5A")
g.polygon([(0,929),(505,686),(662,736),(979,620),(1024,1024),(0,1024)],fill="#172F43")
pix = land.resize((19,19),Image.Resampling.BOX).resize(art.size,Image.Resampling.NEAREST)
photo = Image.new("L",art.size)
ImageDraw.Draw(photo).rounded_rectangle((149,247,785,779),radius=45,fill=255)
field.paste(pix,(0,0),photo)
d.line((674,618,814,769),fill="#161E2B",width=100)
d.line((667,604,820,760),fill="#E8AC70",width=66)
d.ellipse((787,728,853,794),fill="#E8AC70")
d.ellipse((339,175,779,615),fill="#1D303E")
lens = Image.new("L",art.size)
ImageDraw.Draw(lens).ellipse((358,194,760,596),fill=255)
field.paste(land,(0,0),lens)
d.ellipse((335,171,783,619),outline="#F4D7A8",width=25)
art.paste(field,(0,0),mask)
for size in (16,32,128,256,512):
    for scale in (1,2):
        suffix = "@2x" if scale==2 else ""
        art.resize((size*scale,size*scale),Image.Resampling.LANCZOS).save(out/f"icon_{size}x{size}{suffix}.png")
