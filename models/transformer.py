import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torchvision

import models.resnet

class _Block(nn.Module):
    def __init__(self, in_c, out_c):
        super(_Block, self).__init__()

        self.pool = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    def forward(self, x):
        x = self.pool(x)
        return x + self.conv(x)


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encode_channels=128):
        super(Encoder, self).__init__()
        self.c = encode_channels
        self.blocks = nn.ModuleList([
            _Block(1, encode_channels),
            _Block(encode_channels, encode_channels),
            _Block(encode_channels, encode_channels),
            _Block(encode_channels, encode_channels)
        ])

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        features = []
        x = images
        for b in self.blocks:
            x = b(x)
            features.append(x.permute(2, 3, 0, 1).view(x.size(-1) * x.size(-1), -1, self.c))
        # print([f.size() for f in features])
        return torch.cat(features[-2:], dim=0)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=7):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe = torch.zeros(max_len, d_model)
        # position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer('pe', pe)

        self.pe = nn.Parameter(torch.Tensor(max_len, 1, d_model))
        nn.init.normal_(self.pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class CangjieTransformer(nn.Module):
    def __init__(self, codemap, ninp, nhead, nhid, nlayers, dropout=0.2):
        super(CangjieTransformer, self).__init__()
        self.ninp = ninp
        self.codemap = codemap
        ntoken = len(codemap)
        self.encoder = models.resnet.ResNet14()
        # self.transformer = nn.Transformer(ninp, nhead, nlayers, nlayers, nhid, dropout)
        dec_layer = nn.TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer = nn.TransformerDecoder(dec_layer, nlayers)
        self.embedding = nn.Embedding(ntoken, ninp)
        self.ps = PositionalEncoding(ninp)
        self.output = nn.Sequential(
            nn.Linear(ninp, ninp // 2, bias=False),
            nn.ReLU(),
            nn.Linear(ninp // 2, ntoken, bias=False))

        self.aux_codelen = nn.Sequential(
            nn.Linear(ninp, ninp // 2, False),
            nn.BatchNorm1d(ninp // 2),
            nn.ReLU(),
            nn.Linear(ninp // 2, 5, False)
        )


    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, img, tgt):
        src = self.encoder(img)[-1]
        codelen = self.aux_codelen(F.adaptive_avg_pool2d(src, 1).view(-1, src.size(1)))
        src = src.permute(2, 3, 0, 1).view(-1, src.size(0), src.size(1))
        # src = self.ps(src)

        tgt_padding_mask = (tgt == self.codemap['<pad>']).permute(1, 0)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)

        tgt = self.embedding(tgt) * math.sqrt(self.ninp)
        tgt = self.ps(tgt)

        out = self.output(self.transformer(tgt, src, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask))
        # out = self.output(self.transformer(src, tgt, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_padding_mask))
        return out, codelen


