import torch
from einops import rearrange
from omegaconf import OmegaConf
from models.model_utils import freeze_params
from lib.setup_model import setup_model, load_checkpoint


class Representation:
    def __init__(self, exp_params, device):
        self.params = exp_params["representation"]
        self.kind = self.params["type"]
        self.device = device
        self.scale = self.params.get("scale", 1.0)
        self.mean, self.std = 0.0, 1.0
        self.temporal_factor = 1
        if self.kind == "slots":
            self.model = setup_model(exp_params["model"])
            self.model = load_checkpoint(self.params["checkpoint"], self.model, only_model=True, map_cpu=True)
        elif self.kind == "sdvae":
            from diffusers import AutoencoderKL
            self.model = AutoencoderKL.from_pretrained(self.params["model_id"])
        elif self.kind in ("imagevae", "videovae"):
            config = OmegaConf.load(self.params["config"])
            if self.kind == "imagevae":
                from models.autoencoder import ImageVAE
                self.model = ImageVAE(config)
            else:
                from models.video_vae import VideoVAE
                self.model = VideoVAE(**OmegaConf.to_container(config.model, resolve=True))
                self.temporal_factor = self.model.temporal_downsampling_factor
            state = torch.load(self.params["checkpoint"], map_location="cpu", weights_only=False)
            state = {k: v for k, v in state["model_state_dict"].items() if not k.startswith("loss")}
            self.model.load_state_dict(state)
        elif self.kind == "vavae":
            from va_vae.autoencoder import AutoencoderKL
            self.model = AutoencoderKL(embed_dim=32, ch_mult=(1, 1, 2, 2, 4), ckpt_path=self.params["checkpoint"])
            stats = torch.load(self.params["stats"], map_location=device, weights_only=True)
            self.mean, self.std = stats["mean"], stats["std"]
        elif self.kind == "rae":
            from rae.rae import RAE
            self.model = RAE(
                encoder_config_path="facebook/dinov2-with-registers-base",
                encoder_params={"dinov2_path": "facebook/dinov2-with-registers-base", "normalize": True},
                decoder_config_path=self.params["decoder_config"],
                normalization_stat_path=self.params["stats"],
            )
            state = torch.load(self.params["decoder_checkpoint"], map_location="cpu", weights_only=False)
            state = {k.removeprefix("decoder."): v for k, v in state["model"].items() if k.startswith("decoder.")}
            self.model.decoder.load_state_dict(state)
        else:
            raise ValueError(f"Unknown representation: {self.kind}")
        self.model = freeze_params(self.model.to(device).eval())

    @torch.no_grad()
    def encode(self, videos, cached=None):
        if cached is not None:
            return cached.to(self.device) * self.scale if self.kind == "slots" else cached.to(self.device)
        if self.kind == "slots":
            return self.model(videos, num_imgs=videos.shape[1])["slot_history"] * self.scale
        if self.kind == "videovae":
            x = rearrange(2.0 * videos - 1.0, "b t c h w -> b c t h w")
            latents = rearrange(self.model.encode(x).sample(), "b c t h w -> b t c h w")
            return latents[:, -self.frames_to_tokens(videos.shape[1]):] * self.scale
        x = rearrange(videos, "b t c h w -> (b t) c h w")
        if self.kind == "rae":
            x = self.model.encode(x)
        else:
            posterior = self.model.encode(2.0 * x - 1.0)
            if self.kind == "sdvae":
                posterior = posterior.latent_dist
            x = (posterior.sample() - self.mean) / self.std * self.scale
        return rearrange(x, "(b t) c h w -> b t c h w", b=videos.shape[0])

    @torch.no_grad()
    def decode(self, latents, num_frames=None):
        batch, length = latents.shape[:2]
        if self.kind == "slots":
            slots = rearrange(latents / self.scale, "b t s d -> (b t) s d")
            _, output = self.model.decode(slots)
            images = output["recons_img"]
        elif self.kind == "videovae":
            x = rearrange(latents / self.scale, "b t c h w -> b c t h w")
            if not self.model.is_causal:
                padding = (-length) % self.model.temporal_latent_length
                x = torch.cat([x[:, :, :1].expand(-1, -1, padding, -1, -1), x], dim=2)
            images = self.model.decode(x, desired_length=num_frames) * 0.5 + 0.5
            return rearrange(images, "b c t h w -> b t c h w").clamp(0, 1)
        else:
            x = rearrange(latents, "b t c h w -> (b t) c h w")
            if self.kind == "rae":
                images = self.model.decode(x)
            else:
                images = self.model.decode(x / self.scale * self.std + self.mean)
                if self.kind == "sdvae":
                    images = images.sample
                images = images * 0.5 + 0.5
        return rearrange(images, "(b t) c h w -> b t c h w", b=batch, t=length).clamp(0, 1)

    def frames_to_tokens(self, frames):
        return (frames + self.temporal_factor - 1) // self.temporal_factor
