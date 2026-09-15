import torch
from omegaconf import DictConfig
from vae_modules.base_vae import VAE
from vae_modules.distribution import DiagonalGaussianDistribution
from models.image_vae import Encoder, Decoder

class ImageVAE(VAE):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()
        ddconfig, embed_dim = cfg.ddconfig, cfg.embed_dim
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
