import os
from tqdm import tqdm
import torch

from lib.config import Config
from lib.logger import print_, log_function, for_all_methods, log_info
from lib.setup_model import emergency_save
import lib.setup_model as setup_model
import lib.utils as utils
import data as datalib

import torch.distributed as dist

@for_all_methods(log_function)
class BasePredictorTrainerDiffusion:
    def __init__(self, name_predictor_experiment, exp_path, savi_model, checkpoint=None,
                 resume_training=False):
        self.parent_exp_path = exp_path
        self.name_predictor_experiment = name_predictor_experiment
        self.exp_path = os.path.join(exp_path, name_predictor_experiment)
        self.cfg = Config(self.exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
        self.savi_model = savi_model
        self.checkpoint = checkpoint
        self.resume_training = resume_training

        self.plots_path = os.path.join(self.exp_path, "plots", "valid_plots")
        utils.create_directory(self.plots_path)
        self.models_path = os.path.join(self.exp_path, "models")
        utils.create_directory(self.models_path)
        tboard_logs = os.path.join(self.exp_path, "tboard_logs", f"tboard_{utils.timestamp()}")
        utils.create_directory(tboard_logs)

        self.training_losses = []
        self.validation_losses = []
        self.writer = utils.TensorboardWriter(logdir=tboard_logs)
        return

    def load_data(self):
        self.dataset_name = self.exp_params["dataset"]["dataset_name"]
        self.dataset_name = self.dataset_name.split("-")[0]

        batch_size = self.exp_params["training_prediction"]["batch_size"]
        shuffle_train = self.exp_params["dataset"]["shuffle_train"]
        shuffle_eval = self.exp_params["dataset"]["shuffle_eval"]

        train_set = datalib.load_data(exp_params=self.exp_params, split="train")
        print_(f"Examples in training set: {len(train_set)}")
        valid_set = datalib.load_data(exp_params=self.exp_params, split="valid")
        print_(f"Examples in validation set: {len(valid_set)}")

        if hasattr(self, "world_size") and self.world_size > 1:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_set, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle_train
            )
            self.valid_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_set, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle_eval
            )
            self.train_loader = datalib.build_data_loader(
                dataset=train_set,
                batch_size=batch_size,
                shuffle=False,
                sampler=self.train_sampler,
                drop_last=True
            )
            self.valid_loader = datalib.build_data_loader(
                dataset=valid_set,
                batch_size=batch_size,
                shuffle=False,
                sampler=self.valid_sampler,
                drop_last=True
            )
        else:
            self.train_loader = datalib.build_data_loader(
                dataset=train_set,
                batch_size=batch_size,
                shuffle=shuffle_train
            )
            self.valid_loader = datalib.build_data_loader(
                dataset=valid_set,
                batch_size=batch_size,
                shuffle=shuffle_eval
            )
        return


    @emergency_save
    def training_loop(self):
        num_epochs = self.exp_params["training_prediction"]["num_epochs"]
        save_frequency = self.exp_params["training_prediction"]["save_frequency"]

        if self.use_ema:
            self.ema.eval()

        epoch = self.epoch
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            if not hasattr(self, "rank") or self.rank == 0:
                log_info(message=f"Epoch {epoch}/{num_epochs}")
            self.predictor.eval()
            self.valid_epoch(epoch)
            self.predictor.train()
            self.train_epoch(epoch)

            if not hasattr(self, "rank") or self.rank == 0:
                self.writer.add_scalars(
                        plot_name='Total Loss',
                        val_names=["train_loss", "eval_loss"],
                        vals=[self.training_losses[-1], self.validation_losses[-1]],
                        step=epoch+1
                    )

            self.warmup_scheduler(
                    iter=-1,
                    epoch=epoch,
                    exp_params=self.exp_params,
                    end_epoch=True,
                    control_metric=None
                )

            self.wrapper_save_checkpoint(epoch=epoch, savename="checkpoint_last_saved.pth")
            if(epoch % save_frequency == 0 and epoch != 0):
                print_("Saving model checkpoint")
                self.wrapper_save_checkpoint(epoch=epoch, savedir="models")

        print_("Finished training procedure")
        print_("Saving final checkpoint")
        self.wrapper_save_checkpoint(epoch=epoch, finished=True)
        return

    def wrapper_save_checkpoint(self, epoch=None, savedir="models", savename=None, finished=False):
        predictor = self.predictor.module if hasattr(self.predictor, "module") else self.predictor

        if self.use_ema:
            self.save_checkpoint(
                    model=predictor,
                    ema=self.ema,
                    optimizer=self.optimizer,
                    scheduler=self.warmup_scheduler.scheduler,
                    lr_warmup=self.warmup_scheduler.lr_warmup,
                    epoch=epoch,
                    exp_path=self.exp_path,
                    savedir=savedir,
                    savename=savename,
                    finished=finished
                )
        else:
            setup_model.save_checkpoint(
                model=predictor,
                optimizer=self.optimizer,
                scheduler=self.warmup_scheduler.scheduler,
                lr_warmup=self.warmup_scheduler.lr_warmup,
                epoch=epoch,
                exp_path=self.exp_path,
                savedir=savedir,
                savename=savename,
                finished=finished
            )
        return

    @log_function
    def save_checkpoint(self, model, ema, optimizer, scheduler, lr_warmup, epoch, exp_path,
                        finished=False, savedir="models", savename=None):
        if(savename is not None):
            checkpoint_name = savename
        elif(savename is None and finished is True):
            checkpoint_name = "checkpoint_epoch_final.pth"
        else:
            checkpoint_name = f"checkpoint_epoch_{epoch}.pth"

        utils.create_directory(exp_path, savedir)
        savepath = os.path.join(exp_path, savedir, checkpoint_name)

        scheduler_data = "" if scheduler is None else scheduler.state_dict()
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "scheduler_state_dict": scheduler_data,
                "lr_warmup": lr_warmup,
                "visual_token_embedding": self.visual_token_embedding.state_dict() if hasattr(self, "visual_token_embedding") else None
            }, savepath)

        return

    def train_epoch(self, epoch):
        if hasattr(self, "train_sampler") and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        max_train_iters = self.exp_params["training_prediction"].get("train_iters_per_epoch", 1e8)
        total_progress_bar = min(len(self.train_loader), max_train_iters)
        progress_bar = tqdm(enumerate(self.train_loader), total=total_progress_bar) if (not hasattr(self, "rank") or self.rank == 0) else enumerate(self.train_loader)

        loss_values = []
        for i, data in progress_bar:
            if i >= max_train_iters:
                break

            iter_ = total_progress_bar * epoch + i
            self.warmup_scheduler(iter=iter_, epoch=epoch, exp_params=self.exp_params, end_epoch=False)

            loss = self.forward_loss_metric(
                    batch_data=data,
                    training=True
                )
            loss_values.append(loss.detach())

            if not hasattr(self, "rank") or self.rank == 0:
                if(iter_ % self.exp_params["training_prediction"]["log_frequency"] == 0):
                    self.writer.log_full_dictionary(
                            dict={"loss": loss},
                            step=iter_,
                            plot_name="Train Loss",
                            dir="Train Loss Iter",
                        )
                    self.writer.add_scalar(
                            name="Learning Rate",
                            val=self.optimizer.param_groups[0]['lr'],
                            step=iter_
                        )

                progress_bar.set_description(f"Epoch {epoch+1} iter {iter_}: train loss {loss.item():.5f}. ")

                if(iter_ % self.exp_params["training_prediction"].get("save_frequency_iters", 1000) == 0):
                    self.wrapper_save_checkpoint(
                            epoch=epoch,
                            savename=f"checkpoint_epoch_{epoch}_iter_{iter_}.pth"
                        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        avg_loss = torch.stack(loss_values).mean()

        if dist.is_available() and dist.is_initialized():
            avg_loss_tensor = avg_loss.clone()
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss_tensor /= dist.get_world_size()
            avg_loss_value = avg_loss_tensor.item()
        else:
            avg_loss_value = avg_loss.item()

        log_info("LOSS VALUE:")
        log_info("--------")
        log_info(f"  Loss:  {round(avg_loss_value, 5)}")
        log_average_loss_value = {"average_loss": avg_loss_value}

        if not hasattr(self, "rank") or self.rank == 0:
            self.writer.log_full_dictionary(
                    dict=log_average_loss_value,
                    step=epoch + 1,
                    plot_name="Train Loss",
                    dir="Train Loss",
                )
            self.training_losses.append(avg_loss_value)

        return

    @torch.no_grad()
    def valid_epoch(self, epoch):
        max_val_iters = self.exp_params["training_prediction"].get("val_iters_per_epoch", 1e8)
        total_progress_bar = min(len(self.valid_loader), max_val_iters)
        progress_bar = tqdm(enumerate(self.valid_loader), total=total_progress_bar) if (not hasattr(self, "rank") or self.rank == 0) else enumerate(self.valid_loader)

        loss_values = []
        for i, data in progress_bar:
            if i >= max_val_iters:
                break
            loss = self.forward_loss_metric(
                    batch_data=data,
                    training=False
                )
            loss_values.append(loss.detach())

            if not hasattr(self, "rank") or self.rank == 0:
                progress_bar.set_description(f"Epoch {epoch+1} iter {i}: valid loss {loss.item():.5f}. ")

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        avg_loss = torch.stack(loss_values).mean()

        if dist.is_available() and dist.is_initialized():
            avg_loss_tensor = avg_loss.clone()
            dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss_tensor /= dist.get_world_size()
            avg_loss_value = avg_loss_tensor.item()
        else:
            avg_loss_value = avg_loss.item()

        log_info("LOSS VALUE:")
        log_info("--------")
        log_info(f"  Loss:  {round(avg_loss_value, 5)}")
        log_average_loss_value = {"average_loss": avg_loss_value}

        if not hasattr(self, "rank") or self.rank == 0:
            self.writer.log_full_dictionary(
                    dict=log_average_loss_value,
                    step=epoch + 1,
                    plot_name="Valid Loss",
                    dir="Valid Loss",
                )
            self.validation_losses.append(avg_loss_value)

        if not hasattr(self, "rank") or self.rank == 0:
            batch_data = next(iter(self.valid_loader))
            self.visualizations(batch_data=batch_data, epoch=epoch)
        return

    def forward_loss_metric(self, batch_data, training=False, inference_only=False, **kwargs):
        raise NotImplementedError("Base Trainer Module does not implement 'forward_loss_metric'...")

    def visualizations(self):
        raise NotImplementedError("Base Trainer Module does not implement 'forward_loss_metric'...")
