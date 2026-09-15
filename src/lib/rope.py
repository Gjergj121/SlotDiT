import typing as tp

from torch import nn
import torch

class XPos(nn.Module):
    def __init__(
        self, dim: int, smoothing: float = 0.4, base_scale: int = 512, device=None, dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        assert dim % 2 == 0
        assert dtype in [torch.float64, torch.float32]
        self.dtype = dtype
        self.base_scale = base_scale

        half_dim = dim // 2
        adim = torch.arange(half_dim, device=device, dtype=dtype)
        decay_rates = (adim / half_dim + smoothing) / (1.0 + smoothing)
        self.register_buffer("decay_rates", decay_rates)
        self.decay: tp.Optional[torch.Tensor] = None

    def get_decay(self, start: int, end: int):
        if self.decay is None or end > self.decay.shape[0]:
            assert isinstance(self.decay_rates, torch.Tensor)
            idx = torch.arange(end, device=self.decay_rates.device, dtype=self.dtype)
            power = idx / self.base_scale
            scale = self.decay_rates ** power.unsqueeze(-1)
            self.decay = torch.polar(scale, torch.zeros_like(scale))
        return self.decay[start:end]

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_period: float = 10000.0,
        xpos: bool = False,
        scale: float = 1.0,
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        assert dim % 2 == 0
        self.scale = scale
        assert dtype in [torch.float64, torch.float32]
        self.dtype = dtype

        adim = torch.arange(0, dim, 2, device=device, dtype=dtype)[: (dim // 2)]
        frequencies = 1.0 / (max_period ** (adim / dim))
        self.register_buffer("frequencies", frequencies)
        self.rotation: tp.Optional[torch.Tensor] = None

        self.xpos = XPos(dim, device=device, dtype=dtype) if xpos else None

    def get_rotation(self, start: int, end: int):
        if self.rotation is None or end > self.rotation.shape[0]:
            assert isinstance(self.frequencies, torch.Tensor)
            idx = torch.arange(end, device=self.frequencies.device, dtype=self.dtype)
            angles = torch.outer(idx, self.frequencies)
            self.rotation = torch.polar(torch.ones_like(angles), angles)
        return self.rotation[start:end]

    def rotate(self, x: torch.Tensor, start: int = 0, time_dim: int = 1, invert_decay: bool = False):
        T = x.shape[time_dim]
        target_shape = [1] * x.dim()
        target_shape[time_dim] = T
        target_shape[-1] = -1
        rotation = self.get_rotation(start, start + T).view(target_shape)

        if self.xpos:
            decay = self.xpos.get_decay(start, start + T).view(target_shape)
        else:
            decay = 1.0

        if invert_decay:
            decay = decay**-1

        x_complex = torch.view_as_complex(x.to(self.dtype).reshape(*x.shape[:-1], -1, 2))
        scaled_rotation = (rotation * decay) * self.scale + (1.0 - self.scale)
        x_out = torch.view_as_real(x_complex * scaled_rotation).view_as(x)

        return x_out.type_as(x)

    def rotate_qk(self, query: torch.Tensor, key: torch.Tensor, start: int = 0, time_dim: int = 1):
        query_timesteps = query.shape[time_dim]
        key_timesteps = key.shape[time_dim]
        streaming_offset = key_timesteps - query_timesteps

        query_out = self.rotate(query, start + streaming_offset, time_dim)
        key_out = self.rotate(key, start, time_dim, invert_decay=True)

        return query_out, key_out
