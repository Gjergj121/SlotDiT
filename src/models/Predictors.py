import torch
import torch.nn as nn
from lib.logger import print_
import numpy as np
from einops import rearrange, repeat, reduce
from tqdm import tqdm
class PredictorWrapper(nn.Module):
    def __init__(self, exp_params, predictor):
        super().__init__()
        self.exp_params = exp_params
        self.predictor = predictor
        self.num_context = exp_params['training_prediction']['num_context']
        self.num_preds = exp_params['training_prediction']['num_preds']
        self.predictor_name = exp_params['model']['predictor']['predictor_name']
        self.buffer_size = exp_params['model']['predictor'][self.predictor_name]['buffer_size']

    def forward(self, slot_history, num_preds=None, num_context=None, history_context=4, **kwargs):
        if kwargs.get('training', True):
            return self.forward_dfot(slot_history, **kwargs)
        return self.forward_dfot_inference(slot_history, num_preds=num_preds, num_context=num_context, history_context=history_context, **kwargs)

    def forward_dfot(self, slot_history, **kwargs):
        timestep = kwargs["timestep"]
        y = kwargs["token_embeddings"]
        mask = kwargs["mask"]

        diffusion = kwargs["diffusion"]

        model_kwargs = {
                "y": y,
                "mask": mask
            }
        cur_loss_dict = diffusion.training_losses(self.predictor, slot_history, timestep, model_kwargs)
        cur_loss = cur_loss_dict["loss"].mean()

        return [cur_loss]

    def _extend_x_dim(self, x: torch.Tensor, x_shape) -> torch.Tensor:
        return rearrange(x, "... -> ..." + " 1" * len(x_shape))

    def _extend(self, a: torch.Tensor, x: torch.Tensor):
        return rearrange(a, "... -> ..." + " 1" * (x.ndim - a.ndim))

    def _generate_scheduling_matrix(self, diffusion, horizon):
        scheduling_matrix = np.arange(diffusion.sampling_timesteps, -1, -1)[:, None].repeat(horizon, axis=1)
        scheduling_matrix = torch.from_numpy(scheduling_matrix).long()
        return diffusion.ddim_idx_to_noise_level(scheduling_matrix)

    def prepare(
        self,
        diffusion,
        x: torch.Tensor,
        from_noise_levels: torch.Tensor,
        to_noise_levels: torch.Tensor,
        context_mask: torch.Tensor
    ):
        x = repeat(x, "b t ... -> b h t ...", h=2).clone()
        from_noise_levels = repeat(from_noise_levels, "b t -> b h t", h=2).clone()
        to_noise_levels = repeat(to_noise_levels, "b t -> b h t", h=2).clone()

        from_noise_levels[:, 0, :] = torch.where(
            context_mask >= 1,
            diffusion.timesteps - 1,
            from_noise_levels[:, 0, :],
        )
        to_noise_levels[:, 0, :] = torch.where(
            context_mask >= 1,
            diffusion.timesteps - 1,
            to_noise_levels[:, 0, :],
        )

        x[:, 0, :] = torch.where(
            self._extend(context_mask >= 1, x[:, 0, :]),
            diffusion.q_sample(x[:, 0, :], from_noise_levels[:, 0, :]),
            x[:, 0, :],
        )
        x, from_noise_levels, to_noise_levels = map(
            lambda y: rearrange(y, "b h t ... -> (b h) t ..."),
            (x, from_noise_levels, to_noise_levels),
        )

        return x, from_noise_levels, to_noise_levels

    def compose(self, x: torch.Tensor, guidance_scale) -> torch.Tensor:
        x = rearrange(x, "(b h) t ... -> b h t ...", h=2).clone()
        return x[:, 1, :] * guidance_scale - x[:, 0, :] * (guidance_scale - 1)

    def forward_dfot_inference(self, slot_history, num_preds=None, num_context=None, history_context=1, **kwargs):
        num_preds = num_preds if num_preds is not None else self.num_preds
        num_context = num_context if num_context is not None else self.num_context
        if num_preds < 1 or not 0 < num_context < self.buffer_size:
            raise ValueError("Need positive predictions and context shorter than the diffusion window")
        if not 0 < history_context < self.buffer_size:
            raise ValueError("History must be positive and shorter than the diffusion window")

        B, L, *latent_shape = slot_history.shape

        y = kwargs["token_embeddings"]
        mask = kwargs["mask"]

        using_cfg = kwargs["using_cfg"]
        cfg_scale = kwargs["cfg_scale"]
        diffusion = kwargs["diffusion"]

        if using_cfg:
            y_null = self.predictor.y_embedder.y_embedding[None].repeat(B, 1, 1)
            y = torch.stack([y_null, y], dim=1).flatten(0, 1)
            mask = mask.repeat_interleave(2, dim=0) if mask is not None else None

        sample_fn = self.predictor.forward

        model_kwargs = {
                "y": y,
                "mask": mask,
            }

        history = slot_history[:, :num_context]
        context_mask = torch.zeros((B, self.buffer_size), device=slot_history.device)
        context_mask[:, :num_context] = 1

        generated_frames = 0

        pred_denoisings = []
        while generated_frames < num_preds:
            frames_to_generate = min(num_preds - generated_frames, self.buffer_size - num_context)
            generated_frames += frames_to_generate
            z = torch.randn(B, self.buffer_size, *latent_shape, device=slot_history.device)
            z = torch.clamp(z, -diffusion.clip_noise, diffusion.clip_noise)

            z[:, :num_context] = torch.where(self._extend_x_dim(context_mask, z.shape[2:])[:, :num_context] >= 1, history, z[:, :num_context])

            scheduling_matrix = self._generate_scheduling_matrix(diffusion, self.buffer_size)
            scheduling_matrix = scheduling_matrix.to(slot_history.device)
            scheduling_matrix = repeat(scheduling_matrix, "m t -> m b t", b=B)

            scheduling_matrix = torch.where(
                context_mask[None] >= 1, -1, scheduling_matrix
            )

            diff = scheduling_matrix[1:] - scheduling_matrix[:-1]
            skip = torch.argmax((~reduce(diff == 0, "m b t -> m", torch.all)).float())
            scheduling_matrix = scheduling_matrix[skip:]

            pbar = tqdm(
                    total=scheduling_matrix.shape[0] - 1,
                    initial=0,
                    desc="Sampling with DFoT",
                    leave=False,
                )

            for m in range(scheduling_matrix.shape[0] - 1):
                from_noise_levels = scheduling_matrix[m]
                to_noise_levels = scheduling_matrix[m + 1]

                context_mask = torch.where(
                    torch.logical_and(context_mask == 0, from_noise_levels == -1),
                    2,
                    context_mask,
                )

                xs_pred_prev = z.clone()

                if using_cfg:
                    z, from_noise_levels, to_noise_levels = self.prepare(diffusion, z, from_noise_levels, to_noise_levels, context_mask)

                z = diffusion.sample_step(
                    sample_fn,
                    z,
                    from_noise_levels,
                    to_noise_levels,
                    model_kwargs=model_kwargs,
                    guidance_fn=None,
                    sampling_method="ddim"
                )

                if using_cfg:
                    z = self.compose(z, cfg_scale)

                z = torch.where(
                    self._extend_x_dim(context_mask, z.shape[2:]) == 0, z, xs_pred_prev
                )
                pbar.update(1)

            pbar.close()
            samples = z[:, num_context:]
            pred_denoisings.append(samples)
            num_context = history_context

            history = z[:, -num_context:]

            context_mask = torch.zeros((B, self.buffer_size), device=slot_history.device)

            context_mask[:, :num_context] = 1

        pred_denoisings = torch.cat(pred_denoisings, dim=1)
        pred_denoisings = pred_denoisings[:, :num_preds]
        return pred_denoisings
