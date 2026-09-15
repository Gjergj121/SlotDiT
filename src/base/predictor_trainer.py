from copy import deepcopy
import torch
from base.basePredictorTrainerDiffusion import BasePredictorTrainerDiffusion
from data.load_data import unwrap_batch_data
from lib import setup_model, utils
from lib.representations import Representation
from lib.runtime import text_and_diffusion, embed_text
from lib.schedulers import WarmupVSScehdule


class Trainer(BasePredictorTrainerDiffusion):
    def __init__(self, *args, rank=0, world_size=1, device="cpu", **kwargs):
        super().__init__(*args, **kwargs)
        self.rank, self.world_size, self.device = rank, world_size, torch.device(device)

    def setup_predictor(self, predictor=None):
        is_diffusion = predictor is None
        self.representation = Representation(self.exp_params, self.device)
        predictor = (setup_model.setup_predictor(self.exp_params) if is_diffusion else predictor).to(self.device)
        name = self.exp_params["model"]["predictor"]["predictor_name"]
        self.use_ema = self.exp_params["model"]["predictor"][name]["use_ema"]
        self.ema = deepcopy(predictor).eval() if self.use_ema else None
        if self.ema is not None:
            utils.requires_grad(self.ema, False)
        optimizer, scheduler, warmup = setup_model.setup_optimizer(self.exp_params, predictor, section="training_prediction")
        self.epoch = 0
        if self.checkpoint:
            state = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
            predictor.load_state_dict(state["model_state_dict"])
            if self.ema is not None:
                self.ema.load_state_dict(state.get("ema", state["model_state_dict"]))
            if self.resume_training:
                optimizer.load_state_dict(state["optimizer_state_dict"])
                if scheduler is not None:
                    scheduler.load_state_dict(state["scheduler_state_dict"])
                warmup = state["lr_warmup"]
                self.epoch = state["epoch"] + 1
        if is_diffusion:
            self.text_embedder, self.diffusion = text_and_diffusion(self.exp_params, self.device)
        if self.rank == 0:
            utils.log_architecture(predictor, exp_path=self.exp_path)
        if self.world_size > 1:
            predictor = torch.nn.parallel.DistributedDataParallel(
                predictor, device_ids=[self.device.index] if self.device.type == "cuda" else None,
                find_unused_parameters=True)
        self.predictor, self.optimizer, self.scheduler = predictor, optimizer, scheduler
        self.warmup_scheduler = WarmupVSScehdule(optimizer=optimizer, lr_warmup=warmup, scheduler=scheduler, section="training_prediction")

    def wrapper_save_checkpoint(self, *args, **kwargs):
        if self.rank == 0:
            super().wrapper_save_checkpoint(*args, **kwargs)

    def forward_loss_metric(self, batch_data, training=False, **kwargs):
        videos, _, _, others = unwrap_batch_data(self.exp_params, batch_data)
        videos = videos.to(self.device)
        x = self.representation.encode(videos, cached=others.get("slots"))
        y, mask = embed_text(self.text_embedder, others, self.device)
        timesteps = torch.randint(0, self.diffusion.num_timesteps, x.shape[:2], device=self.device)
        losses = self.predictor(x, timestep=timesteps, token_embeddings=y, mask=mask,
                                diffusion=self.diffusion, training=True)
        loss = torch.stack(losses).mean()
        if training:
            self.optimizer.zero_grad()
            loss.backward()
            if self.exp_params["training_prediction"]["gradient_clipping"]:
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(),
                    self.exp_params["training_prediction"]["clipping_max_value"])
            self.optimizer.step()
            if self.use_ema:
                module = self.predictor.module if self.world_size > 1 else self.predictor
                utils.update_ema(self.ema, module)
        return loss

    @torch.no_grad()
    def visualizations(self, batch_data, epoch):
        return
