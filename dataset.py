import os
import shutil
import glob
import torch
import torchvision.transforms as T
import numpy as np

from torch.nn.utils.rnn import pad_sequence
from PIL import ImageFont, ImageDraw, Image
from fontTools.ttLib import TTFont

import utils


def pad_mask(mask, fontsize):
    padded_mask = []
    for l in mask:
        padded_mask.append(l.tolist() + [0] * (fontsize + 1 - len(l)))
    for i in range(fontsize + 1 - len(padded_mask)):
        padded_mask.append([0]*(fontsize + 1))
    return np.array(padded_mask)[: fontsize + 1, : fontsize + 1]

def render_text(text, font):
    mask = font.getmask(text)
    size = mask.size[::-1]
    a = np.asarray(mask).reshape(size)
    return a

class Glyph(object):
    def __init__(self, fonts, size=64):
        self.codepoints = [set(), set()]
        self.size = int(size * 0.8)
        self.size_img = size
        self.pad = (size - self.size) // 2
        self.fonts = [ImageFont.truetype(f, self.size) for f in fonts]
        self.cache = {}
        for cp, font in zip(self.codepoints, fonts):
            font = TTFont(font)
            for cmap in font['cmap'].tables:
                if not cmap.isUnicode():
                    continue
                for k in cmap.cmap:
                    cp.add(k)
    
    def draw(self, ch):
        if ch in self.cache:
            return self.cache[ch]
        if ord(ch) in self.codepoints[0]:
            font = self.fonts[0]
        elif ord(ch) in self.codepoints[1]:
            font = self.fonts[1]
        else:
            return None

        img = Image.new('L', (self.size_img, self.size_img), 0)
        draw = ImageDraw.Draw(img)
        (width, baseline), (offset_x, offset_y) = font.font.getsize(ch)
        draw.text((self.pad - offset_x, self.pad - offset_y), ch, font=font, fill=255, stroke_fill=255) 
        self.cache[ch] = img

        return img




class CodeTableDataset(object):
    class _GlyphTransform(object):
        def __init__(self, dset, t):
            self.dset = dset
            self.t = t

        def __enter__(self):
            self.dset.img_transform = self.t
        
        def __exit__(self, type, value, traceback):
            self.dset.img_transform = T.ToTensor()

    def __init__(self, glyph: Glyph, table: str, codemap: str):
        dset = self

        self.codemap, self.codemap_rev = utils.load_map(codemap)
        self.code_num = len(self.codemap)
        self.glyph = glyph
        self.img_transform = T.ToTensor()

        with open(table) as f:
            entries = [datum.strip().split('\t') for datum in f.readlines()]
            self.entries = [(datum[0], datum[1]) for datum in entries if len(datum) > 1]

        self.chs = []
        self.chs_rev = {}
        for ch, _ in self.entries:
            if ch not in self.chs_rev:
                self.chs_rev[ch] = len(self.chs)
                self.chs.append(ch)

    def __getitem__(self, idx):
        ch, code = self.entries[idx]
        img = self.glyph.draw(ch) 
        if img is None:
            return None

        code_tensor = torch.tensor([self.codemap['<start>']] + [self.codemap[c] for c in code] + [self.codemap['<end>']], dtype=torch.long)

        codelen = torch.tensor(code_tensor.size(0), dtype=torch.long)

        return self.img_transform(img), code_tensor, codelen, torch.tensor(self.chs_rev[ch], dtype=torch.long)

    def __len__(self):
        return len(self.entries)

    @property
    def char_num(self):
        return len(self.chs)
    
    def transform(self, t):
        return self._GlyphTransform(self, t)


def collate_batch(batch):
    batch = [d for d in batch if d is not None]
    img, code, codelen, chidx = zip(*batch)
    return torch.stack(img, dim=0), pad_sequence(code, batch_first=True), torch.stack(codelen, dim=0), torch.stack(chidx, dim=0)
