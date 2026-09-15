import os
import torch

from data.load_data import unwrap_batch_data
from lib.runtime import training_arguments, distributed_setup
from lib.logger import Logger, print_
import lib.utils as utils
from models import freeze_params
import lib.setup_model as setup_model
from lib.loss import LossTracker
from lib.schedulers import WarmupVSScehdule

from base.baseTrainerMultiGPU import BaseTrainer


import torch.distributed as dist

class Trainer(BaseTrainer):
    def __init__(self, *args, rank=0, world_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if torch.cuda.is_available() else "cpu")

    def setup_model(self):
        torch.backends.cudnn.fastest = True

        model = setup_model.setup_model(model_params=self.exp_params["model"])
        if self.rank == 0:
            utils.log_architecture(model, exp_path=self.exp_path)
        model = model.eval().to(self.device)

        optimizer, scheduler, lr_warmup = setup_model.setup_optimizer(
                exp_params=self.exp_params,
                model=model
            )
        loss_tracker = LossTracker(loss_params=self.exp_params["loss"], device=self.device)

        epoch = 0

        if self.checkpoint is not None:
            print_(f"Loading pretrained parameters from checkpoint {self.checkpoint}...")
            loaded_objects = setup_model.load_checkpoint(
                    checkpoint_path=os.path.join(self.models_path, self.checkpoint),
                    model=model,
                    only_model=not self.resume_training,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    lr_warmup=lr_warmup
                )
            if self.resume_training:
                model, optimizer, scheduler, lr_warmup, epoch = loaded_objects
                print_(f"Resuming training from epoch {epoch}...")
            else:
                model = loaded_objects

        freeze_params(model.encoder)

        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.device.index] if self.device.type == "cuda" else None, find_unused_parameters=True
            )

        self.model = model
        self.optimizer, self.scheduler, self.epoch = optimizer, scheduler, epoch
        self.loss_tracker = loss_tracker
        self.warmup_scheduler = WarmupVSScehdule(
                optimizer=self.optimizer,
                lr_warmup=lr_warmup,
                scheduler=scheduler
            )

        return

    def wrapper_save_checkpoint(self, *args, **kwargs):
        if self.rank == 0:
            super().wrapper_save_checkpoint(*args, **kwargs)

    def forward_loss_metric(self, batch_data, training=False, inference_only=False, **kwargs):
        videos, targets, initializer_kwargs, others = unwrap_batch_data(self.exp_params, batch_data)

        batch_size, num_imgs, num_channels, height, width = videos.shape

        videos, targets = videos.to(self.device), targets.to(self.device)
        features = others.pop("features", None)
        if features is None:
            out_model = self.model(
                videos,
                num_imgs=num_imgs,
                **initializer_kwargs
            )
        else:
            features = features.to(self.device)
            out_model = self.model(
                features,
                num_imgs=num_imgs,
                **initializer_kwargs
            )

        if inference_only:
            return out_model, None

        self.loss_tracker(
                preds_feats= out_model.pop("recons_feats"),
                targets_feats= out_model.pop("encoded_img_feats"),
                pred_imgs= out_model.pop("recons_img").clamp(0, 1) if out_model["recons_img"] is not None else out_model.pop("recons_img"),
                target_imgs= targets.clamp(0, 1),
                predicted_temp_probs= out_model.get("predicted_temp_probs", torch.Tensor([])),
                target_temp_probs= out_model.get("target_temp_probs", torch.Tensor([]))
            )

        loss = self.loss_tracker.get_last_losses(total_only=True)

        if training:
            self.optimizer.zero_grad()
            loss.backward()
            if self.exp_params["training_slots"]["gradient_clipping"]:
                torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.exp_params["training_slots"]["clipping_max_value"]
                    )
            self.optimizer.step()

        return out_model, loss

    def visualizations(self, batch_data, epoch):
        return


def main():
    args = training_arguments()
    rank, world_size, device = distributed_setup()
    if rank == 0:
        Logger(args.exp_directory)
    trainer = Trainer(exp_path=args.exp_directory, checkpoint=args.checkpoint,
                      resume_training=args.resume_training, rank=rank, world_size=world_size)
    trainer.device = device
    trainer.exp_params["training_prediction"]["sample_length"] = 5
    for key in ("slots_path", "latents_path", "precomputed_slots_path", "precomputed_latents_path"):
        trainer.exp_params["dataset"].pop(key, None)
    try:
        trainer.load_data()
        trainer.setup_model()
        trainer.training_loop()
    finally:
        trainer.writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
