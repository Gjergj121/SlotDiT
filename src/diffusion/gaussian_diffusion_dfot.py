import math

import torch
import numpy as np
import torch as th
import enum
from torch.nn import functional as F
from einops import rearrange, reduce
from collections import namedtuple
from typing import Optional, Callable

from lib.logger import print_
from diffusion.diffusion_utils import discretized_gaussian_log_likelihood, normal_kl

ModelPrediction = namedtuple(
    "ModelPrediction", ["pred_noise", "pred_x_start", "model_out"]
)

def mean_flat(tensor):


    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class ModelMeanType(enum.Enum):


    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


class ModelVarType(enum.Enum):


    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = (
        enum.auto()
    )
    KL = enum.auto()
    RESCALED_KL = enum.auto()

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):


    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, shift=1.0, zero_terminal_snr=True):


    if schedule_name == "linear":


        scale = 1000 / num_diffusion_timesteps
        betas = get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
        return betas
    elif schedule_name == "squaredcos_cap_v2":
        alphas_cumprod = betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "cosine":
        print_(f"    --> Using cosine beta scheduling with shift={shift}, zero_terminal_snr={zero_terminal_snr}")
        alphas_cumprod = betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.0001) / 1.0001 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


    if zero_terminal_snr:
        alphas_cumprod = enforce_zero_terminal_snr(alphas_cumprod)


    if shift != 1.0:
        alphas_cumprod = shift_alphas_cumprod(alphas_cumprod, shift)


    alphas = np.concatenate([[alphas_cumprod[0]], alphas_cumprod[1:] / alphas_cumprod[:-1]])
    betas = 1 - alphas
    return betas

def shift_alphas_cumprod(alphas_cumprod, shift):


    snr_scale = shift ** 2
    
    alphas_cumprod_shifted = (snr_scale * alphas_cumprod) / (
        snr_scale * alphas_cumprod + 1 - alphas_cumprod
    )
    
    return alphas_cumprod_shifted

def enforce_zero_terminal_snr(alphas_cumprod):


    alphas_cumprod_sqrt = np.sqrt(alphas_cumprod)


    alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].copy()
    alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].copy()


    alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T


    alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / (alphas_cumprod_sqrt[0] + 1e-8)


    alphas_cumprod = alphas_cumprod_sqrt ** 2


    assert np.abs(alphas_cumprod[-1]) < 1e-6, f"terminal SNR not zero: {alphas_cumprod[-1]}"
    
    return alphas_cumprod

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):


    alphas_cumprod = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        alphas_cumprod.append(alpha_bar(t2) / alpha_bar(t1))
    alphas_cumprod = np.cumprod(alphas_cumprod)
    return alphas_cumprod


class GaussianDiffusion:


    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type
    ):
        
        self.clip_noise = 20.0

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type


        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        self.timesteps = self.num_timesteps
        self.sampling_timesteps = 50

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)


        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(np.maximum(1.0 - self.alphas_cumprod, 1e-20))
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / np.maximum(self.alphas_cumprod, 1e-20))
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(np.maximum(1.0 / self.alphas_cumprod - 1, 0))


        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / np.maximum(1.0 - self.alphas_cumprod, 1e-20)
        )

        self.posterior_log_variance_clipped = np.log(
            np.maximum(np.append(self.posterior_variance[1], self.posterior_variance[1:]), 1e-20)
        ) if len(self.posterior_variance) > 1 else np.array([])

        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / np.maximum(1.0 - self.alphas_cumprod, 1e-20)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / np.maximum(1.0 - self.alphas_cumprod, 1e-20)
        )


        self.snr_clip = 5.0
        self.snr = torch.from_numpy(self.alphas_cumprod / np.maximum(1 - self.alphas_cumprod, 1e-20))
        self.logsnr = torch.log(self.snr)
        self.clipped_snr = self.snr.clone()
        self.clipped_snr.clamp_(max=self.snr_clip)
    
    def ddim_idx_to_noise_level(self, indices: torch.Tensor):
        shape = indices.shape
        real_steps = torch.linspace(-1, self.timesteps - 1, self.sampling_timesteps + 1)
        real_steps = real_steps.long().to(indices.device)
        k = real_steps[indices.flatten()]
        return k.view(shape)

    def q_mean_variance(self, x_start, t):


        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def predict_v(self, x_start, t, noise):
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):


        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, t, model_kwargs=None):


        if model_kwargs is None:
            model_kwargs = {}

        if len(x.shape) == 3:
            B, num_slots, slot_dim = x.shape
        elif len(x.shape) == 4:
            B, T, num_slots, slot_dim = x.shape


        model_output = self.model_predictions(model, x, t, **model_kwargs)
        x_start = model_output.pred_x_start


        return x_start, self.q_posterior(x_start=x_start, x_k=x, k=t)
    
    def q_posterior(self, x_start, x_k, k):
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, k, x_k.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, k, x_k.shape) * x_k
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, k, x_k.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, k, x_k.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    
    def predict_start_from_v(self, x_k, k, v):
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, k, x_k.shape) * x_k
            - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * v
        )

    def predict_noise_from_v(self, x_k, k, v):
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, k, x_k.shape) * v
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, k, x_k.shape) * x_k
        )
    
    def predict_start_from_noise(self, x_k, k, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, k, x_k.shape) * x_k
            - extract(self.sqrt_recipm1_alphas_cumprod, k, x_k.shape) * noise
        )

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        history_condition=None,
        num_context=0,
    ):


        original_batch_size = shape[0] // 2 if history_condition is not None else shape[0]
        if device is None:
            device = next(model.parameters()).device


        img = torch.randn(*shape, device=device) if noise is None else noise


        indices = torch.arange(self.num_timesteps, 0, -1, device=device)

        if progress:
            from tqdm.auto import tqdm
            indices_progress = tqdm(indices.tolist())
        else:
            indices_progress = indices.tolist()

        for i in indices_progress:
            t_curr = torch.tensor([i - 1] * shape[0], device=device)
            t_next = torch.tensor([i - 2] * shape[0], device=device)


            if history_condition is not None:
                assert num_context > 0, "num_context must be > 0 for history conditioning"


                noised_history = self.q_sample(history_condition, t_curr)


                img[:, :num_context, :, :] = history_condition


            x_start, out = self.p_mean_variance(
                model, img, t_curr, clip_denoised, denoised_fn, model_kwargs=model_kwargs
            )


            eps = self._predict_eps_from_xstart(img, t_curr, x_start)
            alpha_bar = _extract_into_tensor(self.alphas_cumprod, t_curr, img.shape)
            alpha_bar_next = torch.where(
                (t_next < 0).view(-1, 1, 1, 1).expand_as(alpha_bar),
                torch.ones_like(alpha_bar),
                _extract_into_tensor(self.alphas_cumprod, t_next, img.shape)
            )
            sigma = eta * torch.sqrt((1 - alpha_bar_next) / (1 - alpha_bar)) * torch.sqrt(1 - alpha_bar / alpha_bar_next)
            mean_pred = x_start * torch.sqrt(alpha_bar_next) + torch.sqrt(1 - alpha_bar_next - sigma**2) * eps

            rand_noise = torch.randn_like(img)
            nonzero_mask = (t_curr != 0).float().view(-1, *([1] * (len(img.shape) - 1)))
            sample = mean_pred + nonzero_mask * sigma * rand_noise

            img = sample

        return img
    
    def sample_step(
        self,
        model,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        model_kwargs=None,
        guidance_fn: Optional[Callable] = None,
        sampling_method: str = "ddim",
    ):
        if sampling_method == "ddim":
            return self.ddim_sample_step(
                model=model,
                x=x,
                curr_noise_level=curr_noise_level,
                next_noise_level=next_noise_level,
                model_kwargs=model_kwargs,
                guidance_fn=guidance_fn,
            )


        assert torch.all(
            (curr_noise_level - 1 == next_noise_level)
            | ((curr_noise_level == -1) & (next_noise_level == -1))
        ), "Wrong noise level given for ddpm sampling."

        assert (
            self.sampling_timesteps == self.timesteps
        ), "sampling_timesteps should be equal to timesteps for ddpm sampling."

        return self.ddpm_sample_step(
            model=model,
            x=x,
            curr_noise_level=curr_noise_level,
            model_kwargs=model_kwargs,
            guidance_fn=guidance_fn,
        )

    def ddpm_sample_step(
        self,
        model,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        model_kwargs=None,
        guidance_fn: Optional[Callable] = None,
    ):
        if guidance_fn is not None:
            raise NotImplementedError("guidance_fn is not yet implmented for ddpm.")

        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)

        _, model_mean, _, model_log_variance = self.p_mean_variance(
            model=model,
            x=x,
            t=clipped_curr_noise_level,
            model_kwargs=model_kwargs,
        )

        noise = torch.where(
            self.add_shape_channels(clipped_curr_noise_level > 0, x.shape[2:]),
            torch.randn_like(x),
            0,
        )
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        x_pred = model_mean + torch.exp(0.5 * model_log_variance) * noise


        return torch.where(self.add_shape_channels(curr_noise_level == -1, x.shape[2:]), x, x_pred)

    def ddim_sample_step(
        self,
        model,
        x: torch.Tensor,
        curr_noise_level: torch.Tensor,
        next_noise_level: torch.Tensor,
        model_kwargs=None,
        guidance_fn: Optional[Callable] = None,
    ):
        ddim_sampling_eta = 0.0
        clipped_curr_noise_level = torch.clamp(curr_noise_level, min=0)

        alpha = th.from_numpy(self.alphas_cumprod).to(device=clipped_curr_noise_level.device)[clipped_curr_noise_level].float()
        alpha_next = torch.where(
            next_noise_level < 0,
            torch.ones_like(next_noise_level),

            th.from_numpy(self.alphas_cumprod).to(device=next_noise_level.device)[next_noise_level].float(),
        )
        sigma = torch.where(
            next_noise_level < 0,
            torch.zeros_like(next_noise_level),
            ddim_sampling_eta
            * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt(),
        )
        c = (1 - alpha_next - sigma**2).sqrt()

        alpha = self.add_shape_channels(alpha, x.shape[2:])
        alpha_next = self.add_shape_channels(alpha_next, x.shape[2:])
        c = self.add_shape_channels(c, x.shape[2:])
        sigma = self.add_shape_channels(sigma, x.shape[2:])

        if guidance_fn is not None:
            with torch.enable_grad():
                x = x.detach().requires_grad_()

                model_pred = self.model_predictions(
                    model,
                    x,
                    clipped_curr_noise_level,
                    **model_kwargs,
                )

                guidance_loss = guidance_fn(
                    xk=x, pred_x0=model_pred.pred_x_start, alpha_cumprod=alpha
                )

                grad = -torch.autograd.grad(
                    guidance_loss,
                    x,
                )[0]
                grad = torch.nan_to_num(grad, nan=0.0)

                pred_noise = model_pred.pred_noise + (1 - alpha).sqrt() * grad
                x_start = torch.where(
                    alpha > 0,
                    self.predict_start_from_noise(
                        x, clipped_curr_noise_level, pred_noise
                    ),
                    model_pred.pred_x_start,
                )

        else:
            model_pred = self.model_predictions(
                model,
                x,
                clipped_curr_noise_level,
                **model_kwargs,
            )
            x_start = model_pred.pred_x_start
            pred_noise = model_pred.pred_noise

        noise = torch.randn_like(x)
        noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        x_pred = x_start * alpha_next.sqrt() + pred_noise * c + sigma * noise


        mask = curr_noise_level == next_noise_level
        x_pred = torch.where(
            self.add_shape_channels(mask, x.shape[2:]),
            x,
            x_pred,
        )

        return x_pred
    
    def q_sample(self, x_start, t, noise=None):


        if noise is None:
            noise = th.randn_like(x_start)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)
        assert noise.shape == x_start.shape
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, conditioning_mask=None, cond_indicator=None):


        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
            noise = torch.clamp(noise, -self.clip_noise, self.clip_noise)

        x_t = self.q_sample(x_start, t, noise=noise)


        if cond_indicator is not None:
            t = t.unsqueeze(-1) * (1 - cond_indicator)
            t = t.long()
            model_kwargs['condition_mask'] = conditioning_mask
        
        model_output = self.model_predictions(model, x_t, t, **model_kwargs)

        pred = model_output.model_out
        x_pred = model_output.pred_x_start


        target = self.predict_v(x_start, t, noise)


        terms = {}
        loss = F.mse_loss(pred, target.detach(), reduction="none")
        if t.dim() == 1:
            t = t.unsqueeze(1).repeat(1, x_start.shape[1])
        loss_weight = self.compute_loss_weights(t, "fused_min_snr")
        loss_weight = self.add_shape_channels(loss_weight, pred.shape[2:])
        loss = loss * loss_weight
        loss = mean_flat(loss)

        terms["mse"] = loss
        terms["loss"] = terms["mse"]

        return terms

    def model_predictions(self, model, x, k, **model_kwargs):


        model_output = model(x, k, **model_kwargs)


        v = model_output
        x_start = self.predict_start_from_v(x, k, v)
        pred_noise = self.predict_noise_from_v(x, k, v)

        model_pred = ModelPrediction(pred_noise, x_start, model_output)

        return model_pred

    def compute_loss_weights(self, k, strategy) -> torch.Tensor:


        if strategy == "uniform":
            return torch.ones_like(k)
        self.snr = self.snr.to(k.device)
        snr = self.snr[k]
        epsilon_weighting = None
        if strategy == "sigmoid":
                logsnr = self.logsnr[k]


                epsilon_weighting = torch.sigmoid(
                    self.cfg.loss_weighting.sigmoid_bias - logsnr
                )
        elif strategy == "min_snr":

            clipped_snr = self.clipped_snr[k]
            epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)
        elif strategy == "fused_min_snr":


            snr_clip = self.snr_clip
            cum_snr_decay = 0.9

            self.clipped_snr = self.clipped_snr.to(k.device)
            clipped_snr = self.clipped_snr[k]
            normalized_clipped_snr = clipped_snr / snr_clip
            normalized_snr = snr / snr_clip

            def compute_cum_snr(reverse: bool = False):
                new_normalized_clipped_snr = (
                    normalized_clipped_snr.flip(1)
                    if reverse
                    else normalized_clipped_snr
                )
                cum_snr = torch.zeros_like(new_normalized_clipped_snr)
                for t in range(0, k.shape[1]):
                    if t == 0:
                        cum_snr[:, t] = new_normalized_clipped_snr[:, t]
                    else:
                        cum_snr[:, t] = (
                            cum_snr_decay * cum_snr[:, t - 1]
                            + (1 - cum_snr_decay) * new_normalized_clipped_snr[:, t]
                        )
                cum_snr = F.pad(cum_snr[:, :-1], (1, 0, 0, 0), value=0.0)
                return cum_snr.flip(1) if reverse else cum_snr


            cum_snr = compute_cum_snr(reverse=True) + compute_cum_snr()
            cum_snr *= 0.5
            clipped_fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (
                1 - normalized_clipped_snr
            )
            fused_snr = 1 - (1 - cum_snr * cum_snr_decay) * (1 - normalized_snr)
            clipped_snr = clipped_fused_snr * snr_clip
            snr = fused_snr * snr_clip
            epsilon_weighting = clipped_snr / snr.clamp(min=1e-8)
        else:
            raise ValueError(f"unknown loss weighting strategy {strategy}")


        return epsilon_weighting * snr / (snr + 1)


    def add_shape_channels(self, x, x_shape):
        return rearrange(x, f"... -> ...{' 1' * len(x_shape)}")


def _extract_into_tensor(arr, timesteps, broadcast_shape):


    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res + th.zeros(broadcast_shape, device=timesteps.device)

def extract(a, t, x_shape):
    shape = t.shape
    out = th.from_numpy(a).to(device=t.device)[t].float()
    return out.reshape(*shape, *((1,) * (len(x_shape) - len(shape))))
