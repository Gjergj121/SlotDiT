from typing import Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from vae_modules.distribution import DiagonalGaussianDistribution

class VAE(ABC, nn.Module):
    @abstractmethod
    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        raise NotImplementedError

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self, sample: torch.Tensor, sample_posterior: bool = True
    ) -> Tuple[torch.Tensor, DiagonalGaussianDistribution]:
        posterior = self.encode(sample)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "VAE":
        raise NotImplementedError
