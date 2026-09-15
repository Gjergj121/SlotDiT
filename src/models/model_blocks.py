import math
import torch
import torch.nn as nn

from models.model_utils import build_grid

__all__ = ["ConvBlock", "ConvTransposeBlock", "ResnetBlock", "AttnBlock", "SoftPositionEmbed", "PositionalEncoding", "TemporalPositionalEncoding", "LearnedPositionalEncoding1D"]

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None,
                 batch_norm=False, max_pool=None, activation=True):
        super().__init__()
        padding = padding if padding is not None else kernel_size // 2

        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        if max_pool:
            assert isinstance(max_pool, (int, tuple, list))
            layers.append(nn.MaxPool2d(kernel_size=max_pool, stride=max_pool))
        if activation:
            layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)
        return

    def forward(self, x):
        y = self.block(x)
        return y

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None,
                 batch_norm=False, upsample=None, activation=True, conv_transpose_2d=True):
        super().__init__()
        padding = padding if padding is not None else kernel_size // 2

        layers = []
        if conv_transpose_2d:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding)
            )
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=out_channels))
        if upsample:
            assert isinstance(upsample, (int, tuple, list))
            layers.append(nn.Upsample(scale_factor=upsample))
        if activation:
            layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)
        return

    def forward(self, x):
        y = self.block(x)
        return y

def nonlinearity(x):
        return x*torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)
        k = k.reshape(b,c,h*w)
        w_ = torch.bmm(q,k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)
        h_ = torch.bmm(v,w_)
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution, vmin=-1., vmax=1.):
        super().__init__()
        self.projection = nn.Conv2d(4, hidden_size, kernel_size=1)
        self.grid = build_grid(resolution, vmin=-1., vmax=1.).permute(0, 3, 1, 2)
        return

    def forward(self, inputs, channels_last=True):
        b_size = inputs.shape[0]
        if self.grid.device != inputs.device:
            self.grid = self.grid.to(inputs.device)
        grid = self.grid.repeat(b_size, 1, 1, 1)
        emb_proj = self.projection(grid)
        if channels_last:
            emb_proj = emb_proj.permute(0, 2, 3, 1)
        return inputs + emb_proj

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.view(1, max_len, 1, d_model)
        self.pe = pe
        return

    def forward(self, x, batch_size, num_slots):
        if x.device != self.pe.device:
            self.pe = self.pe.to(x.device)
        cur_seq_len = x.shape[1]
        cur_pe = self.pe.repeat(batch_size, 1, num_slots, 1)[:, :cur_seq_len]
        x = x + cur_pe
        y = self.dropout(x)
        return y

class TemporalPositionalEncoding(nn.Module):
    MODES = ["sinusoid", "learned"]

    def __init__(self, d_model, dropout=0.0, max_len=50, mode="sinusoid"):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown {mode = }. Use one of {self.MODES}...")
        self.mode = mode
        self.d_model = d_model
        self.max_len = max_len
        self.dropout_p = dropout
        self.dropout = nn.Dropout(p=dropout)

        self.pe = self._get_pe()
        return

    def _get_pe(self):
        max_len = self.max_len
        d_model = self.d_model
        if self.mode == "sinusoid":
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)
            pe = pe.view(1, max_len, 1, d_model)
        elif self.mode == "learned":
            scale = d_model ** -0.5
            pe = nn.Parameter(scale * torch.randn(1, max_len, 1, d_model))
        else:
            raise ValueError(f"UPSI! Unknown {self.mode = }. Use one of {self.MODES}...")
        return pe

    def forward(self, x, batch_size, num_slots):
        if x.device != self.pe.device:
            self.pe = self.pe.to(x.device)
        cur_seq_len = x.shape[1]
        cur_pe = self.pe.repeat(batch_size, 1, num_slots, 1)[:, :cur_seq_len]
        cur_pe = torch.flip(cur_pe, dims=(1,))
        x = x + cur_pe
        y = self.dropout(x)
        return y

    def __repr__(self):
        str = ""
        str += "TemporalPositionalEncoding(\n"
        str += f"  (mode) {self.mode})\n"
        str += f"  (max_len) {self.max_len})\n"
        str += f"  (d_model) {self.d_model})\n"
        str += f"  (dropout): Dropout(p={self.dropout_p} inplace=False)\n"
        str += ")"
        return str

class LearnedPositionalEncoding1D(nn.Module):
    def __init__(self, max_len, token_dim, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, token_dim), requires_grad=True)
        nn.init.trunc_normal_(self.pe)
        return

    def forward(self, x, offset=0):
        T = x.shape[1]
        y = self.dropout(x + self.pe[:, offset:offset + T])
        return y
