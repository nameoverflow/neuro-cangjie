import numpy as np

from PIL import ImageFont, ImageDraw, Image
from fontTools.ttLib import TTFont

if __name__ == '__main__':
    with open('Cangjie5/Cangjie5.txt') as f:
        entries = [datum.strip().split('\t') for datum in f.readlines()]
        entries = [(datum[0], datum[1]) for datum in entries if len(datum) > 1]
    
    background = np.ones([1, 299, 299], dtype=float)

    codepoints = [set(), set()]
    fonts = ['hanazono/HanaMinA.ttf', 'hanazono/HanaMinB.ttf']
    for cp, font in zip(codepoints, fonts):
        font = TTFont(font)
        for cmap in font['cmap'].tables:
            if not cmap.isUnicode():
                continue
            for k in cmap.cmap:
                cp.add(k)

    fonts = [ImageFont.truetype(f, 299) for f in fonts]
    fonts[0].getmetrics()

    
    for ch, code in entries:
        try:
            if ord(ch) in codepoints[0]:
                font = fonts[0]
            elif ord(ch) in codepoints[1]:
                font = fonts[1]
            else:
                continue
        
            img = Image.new('1', (64, 64), 1)
            # img = img.convert(1)
            draw = ImageDraw.Draw(img)
            (width, baseline), (offset_x, offset_y) = font.font.getsize(ch)
            draw.text((0,-offset_y), ch, font=font, fill=0, stroke_fill=0) 
            img.save('img/%s.%s.png'%(ch,code))
        except Exception as e:
            pass

