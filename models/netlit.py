from typing import Any
import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F
from torch import nn, optim
from einops import rearrange

from .help_funcs import Attention, PreNorm, PreNorm2, Residual, Residual2, FeedForward, Cross_Attention
from .networks import ChannelAttention


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        # b, n, _, h = *x.shape, self.heads
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
    

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        B_, C_, H_, W_ = x.shape
        x = x.view([B_, C_, H_*W_])
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        x = x.view([B_, C_, H_, W_])
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, softmax=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual2(PreNorm2(dim, Cross_Attention(dim, heads = heads,
                                                        dim_head = dim_head, dropout = dropout,
                                                        softmax=softmax))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, m, mask = None):
        """target(query), memory"""
        B_, C_, H_, W_ = x.shape
        x = x.view([B_, C_, H_*W_])
        for attn, ff in self.layers:
            x = attn(x, m, mask = mask)
            x = ff(x)
        x = x.view([B_, C_, H_, W_])
        x = self.classifier(x)
        return x

class DifferenceBlock(nn.Module):
    def __init__(self, dim, depth=1, heads = 4, dim_head = 64, mlp_dim = 128, dropout = 0.01):
        super().__init__()
        self.encoder = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.positional_encoding = ChannelAttention(dim, dim,)
        self.decoder = TransformerDecoder(dim, depth, heads, dim_head, mlp_dim, dropout)
        
    def forward(self, x_1, x_2):
        enc1 = self.encoder(x_1)
        enc2 = self.encoder(x_2)
        pos = self.positional_encoding(x_1, x_2)
        out = self.decoder(enc1-enc2, pos)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.norm(x)
        return self.act(x)
    

class UpBlock(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels_1+in_channels_2, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

    def forward(self, x_1, x_2):
        x = torch.cat([x_1, x_2], dim=1)
        x = self.conv(x)
        x = F.interpolate(x)
        return self.act(x)
    

class UpBlockBottle(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(x)
        return self.act(x)


class DAHiTra(nn.Module):
    def __init__(self,):
        super().__init__()
    
        encoder_filters = [64, 64, 128, 256,]
        # encoder_filters = [64, 64, 128, 256, 512]
        # decoder_filters = np.asarray([48, 64, 96, 160, 320])

        self.down1 = DownBlock(3, encoder_filters[0])
        self.down2 = DownBlock(encoder_filters[0], encoder_filters[1])
        self.down3 = DownBlock(encoder_filters[1], encoder_filters[2])
        self.down4 = DownBlock(encoder_filters[2], encoder_filters[3])
        # self.down5 = DownBlock(encoder_filters[3], encoder_filters[4])

        self.diff1 = DifferenceBlock(256*256)
        self.diff2 = DifferenceBlock(128*128)
        self.diff3 = DifferenceBlock(64*64)

        self.up1 = UpBlockBottle(in_channels=encoder_filters[-1], out_channels=32)
        self.up3 = UpBlock(in_channels_1=encoder_filters[-2], in_channels_2=32, out_channels=32)
        self.up4 = UpBlock(in_channels_1=encoder_filters[-3], in_channels_2=32, out_channels=32)
        self.up5 = UpBlock(in_channels_1=encoder_filters[-2], in_channels_2=32, out_channels=5)
        # self.up5 = UpBlock(encoder_filters[-5], 5)
    
    def forward(self, x_1, x_2) -> Any:

        enc1_1 = self.down1(x_1)
        enc2_1 = self.down2(enc1_1)
        enc3_1 = self.down3(enc2_1)
        enc4_1 = self.down4(enc3_1)

        enc1_2 = self.down1(x_2)
        enc2_2 = self.down2(enc1_2)
        enc3_2 = self.down3(enc2_2)
        enc4_2 = self.down4(enc3_2)
        
        skip = torch.cat([enc1_1, enc1_2], dim=1)
        diff1 = self.diff1(enc2_1, enc2_2)
        diff2 = self.diff2(enc3_1, enc3_2)
        diff3 = self.diff3(enc4_1, enc4_2)

        dec1 = self.up1(diff3)
        dec2 = self.up2(dec1, diff2)
        dec3 = self.up2(dec2, diff1)
        out = self.up2(dec3, skip)

        return out
    

class DAHiTraLit(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DAHiTra() 

    def forward(self, x_1, x_2):
        return self.model(x_1, x_2)

    def training_step(self, batch, batch_idx):
        x_1 = batch['A']
        x_2 = batch['B']
        y = batch['L']
        output = self(x_1, x_2)
        loss = F.cross_entropy(output, y)
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y) # FIXME: replace loss
        self.log('test_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x) 
        loss = F.cross_entropy(output, y) # FIXME: replace loss
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        # scheduler goes here
        return optimizer