import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from rotary_embedding_torch.rotary_embedding_torch import rotate_half
from diffusers.models.embeddings import TimestepEmbedding

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    if len(timesteps.shape) not in [1, 2]:
        raise ValueError("Timesteps should be a 1D or 2D tensor")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[..., None].float() * emb

    emb = scale * emb

    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[..., half_dim:], emb[..., :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb

class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb

class StochasticUnknownTimesteps(Timesteps):
    def __init__(
        self,
        num_channels: int,
        p: float = 1.0,
    ):
        super().__init__(num_channels)
        self.unknown_token = (
            nn.Parameter(torch.randn(1, num_channels)) if p > 0.0 else None
        )
        self.p = p

    def forward(self, timesteps: torch.Tensor, mask: Optional[torch.Tensor] = None):
        t_emb = super().forward(timesteps)

        if self.p == 0.0:
            return t_emb

        if self.training or self.p == 1.0 or mask is None:
            mask = torch.rand(t_emb.shape[:-1], device=t_emb.device) < self.p
            mask = mask[..., None].expand_as(t_emb)
            return torch.where(mask, self.unknown_token, t_emb)

        mask = mask[..., None].expand_as(t_emb)
        return torch.where(mask, self.unknown_token, t_emb)

class StochasticTimeEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_embed_dim: int,
        use_fourier: bool = False,
        p: float = 0.0,
    ):
        super().__init__()
        self.use_fourier = use_fourier
        if self.use_fourier:
            assert p == 0.0, "Fourier embeddings do not support stochastic timesteps"
        self.timesteps = (
            StochasticUnknownTimesteps(dim, p)
        )
        self.embedding = TimestepEmbedding(dim, time_embed_dim)

    def forward(self, timesteps: torch.Tensor, mask: Optional[torch.Tensor] = None):
        return self.embedding(
            self.timesteps(timesteps)
            if self.use_fourier
            else self.timesteps(timesteps, mask)
        )

def apply_rotary_emb_allegro(x: torch.Tensor, freqs_cis, positions):
    def apply_1d_rope(tokens, pos, cos, sin):
        cos = F.embedding(pos, cos)[:, None, :, :]
        sin = F.embedding(pos, sin)[:, None, :, :]
        x1, x2 = tokens[..., : tokens.shape[-1] // 2], tokens[..., tokens.shape[-1] // 2 :]
        tokens_rotated = torch.cat((-x2, x1), dim=-1)
        return (tokens.float() * cos + tokens_rotated.float() * sin).to(tokens.dtype)

    (t_cos, t_sin), (h_cos, h_sin), (w_cos, w_sin) = freqs_cis
    t, h, w = x.chunk(3, dim=-1)
    t = apply_1d_rope(t, positions[0], t_cos, t_sin)
    h = apply_1d_rope(h, positions[1], h_cos, h_sin)
    w = apply_1d_rope(w, positions[2], w_cos, w_sin)
    x = torch.cat([t, h, w], dim=-1)
    return x

class RotaryEmbeddingND(nn.Module):
    def __init__(
        self,
        dims: Tuple[int, ...],
        sizes: Tuple[int, ...],
        theta: float = 10000.0,
        flatten: bool = True,
    ):
        super().__init__()
        self.n_dims = len(dims)
        self.dims = dims
        self.theta = theta
        self.flatten = flatten

        Colon = slice(None)
        all_freqs = []
        for i, (dim, seq_len) in enumerate(zip(dims, sizes)):
            freqs = self.get_freqs(dim, seq_len)
            all_axis = [None] * len(dims)
            all_axis[i] = Colon
            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice].expand(*sizes, dim))
        all_freqs = torch.cat(all_freqs, dim=-1)
        if flatten:
            all_freqs = rearrange(all_freqs, "... d -> (...) d")
        self.register_buffer("freqs", all_freqs, persistent=False)

    def get_freqs(self, dim: int, seq_len: int) -> torch.Tensor:
        freqs = 1.0 / (
            self.theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        )
        pos = torch.arange(seq_len, dtype=freqs.dtype)
        freqs = einsum("..., f -> ... f", pos, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_shape = x.shape[-2:-1] if self.flatten else x.shape[-self.n_dims - 1 : -1]
        slice_tuple = tuple(slice(0, seq_len) for seq_len in seq_shape)
        freqs = self.freqs[slice_tuple]
        return x * freqs.cos() + rotate_half(x) * freqs.sin()

class RotaryEmbedding1D(RotaryEmbeddingND):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        theta: float = 10000.0,
        flatten: bool = True,
    ):
        super().__init__((dim,), (seq_len,), theta, flatten)

class RotaryEmbedding2D(RotaryEmbeddingND):
    def __init__(
        self,
        dim: int,
        sizes: Tuple[int, int],
        theta: float = 10000.0,
        flatten: bool = True,
    ):
        assert dim % 2 == 0, "RotaryEmbedding2D requires even dim"
        super().__init__((dim // 2,) * 2, sizes, theta, flatten)

class RotaryEmbedding3D(RotaryEmbeddingND):
    def __init__(
        self,
        dim: int,
        sizes: Tuple[int, int, int],
        theta: float = 10000.0,
        flatten: bool = True,
    ):
        assert dim % 2 == 0, "RotaryEmbedding3D requires even dim"
        dim //= 2

        if dim % 3 == 0:
            dims = (dim // 3,) * 3
        elif dim % 3 == 1:
            dims = (dim // 3 + 1, dim // 3, dim // 3)
        elif dim % 3 == 2:
            dims = (dim // 3, dim // 3 + 1, dim // 3 + 1)

        super().__init__(tuple(d * 2 for d in dims), sizes, theta, flatten)
