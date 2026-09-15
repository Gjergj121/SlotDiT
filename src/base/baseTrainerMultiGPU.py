import os
from tqdm import tqdm
import torch

from lib.config import Config
from lib.logger import print_, log_function, for_all_methods, log_info
from lib.setup_model import emergency_save
import lib.setup_model as setup_model
import lib.utils as utils
import data as datalib

@for_all_methods(log_function)
class BaseTrainer:
    def __init__(self, exp_path, checkpoint=None, resume_training=False):
        self.exp_path = exp_path
        self.cfg = Config(exp_path)
        self.exp_params = self.cfg.load_exp_config_file()
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
        batch_size = self.exp_params["training_slots"]["batch_size"]
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

    def setup_model(self):
        raise NotImplementedError

    @emergency_save
    def training_loop(self):
        num_epochs = self.exp_params["training_slots"]["num_epochs"]
        save_frequency = self.exp_params["training_slots"]["save_frequency"]

        epoch = self.epoch
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            log_info(message=f"Epoch {epoch}/{num_epochs}")
            self.model.eval()
            self.valid_epoch(epoch)
            self.model.train()
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
                    control_metric=self.validation_losses[-1]
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
        model = self.model.module if hasattr(self.model, "module") else self.model

        setup_model.save_checkpoint(
                model=model,
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

    def train_epoch(self, epoch):
        if hasattr(self, "train_sampler") and self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

        max_train_iters = self.exp_params["training_slots"].get("train_iters_per_epoch", 1e8)
        self.loss_tracker.reset()
        total_progress_bar = min(len(self.train_loader), max_train_iters)
        progress_bar = tqdm(enumerate(self.train_loader), total=total_progress_bar)

        for i, data in progress_bar:
            if i >= max_train_iters:
                break

            iter_ = len(self.train_loader) * epoch + i
            self.warmup_scheduler(iter=iter_, epoch=epoch, exp_params=self.exp_params, end_epoch=False)

            out_model, loss = self.forward_loss_metric(
                    batch_data=data,
                    training=True
                )

            if not hasattr(self, "rank") or self.rank == 0:
                if(iter_ % self.exp_params["training_slots"]["log_frequency"] == 0):
                    self.writer.log_full_dictionary(
                            dict=self.loss_tracker.get_last_losses(),
                            step=iter_,
                            plot_name="Train Loss",
                            dir="Train Loss Iter",
                        )
                    self.writer.add_scalar(
                            name="Learning Rate",
                            val=self.optimizer.param_groups[0]['lr'],
                            step=iter_
                        )

            progress_bar.set_description(f"Epoch {epoch+1} iter {i}: train loss {loss.item():.5f}. ")

        self.loss_tracker.aggregate()
        average_loss_vals = self.loss_tracker.summary(log=True, get_results=True)
        if not hasattr(self, "rank") or self.rank == 0:
            self.writer.log_full_dictionary(
                    dict=average_loss_vals,
                    step=epoch + 1,
                    plot_name="Train Loss",
                    dir="Train Loss",
                )
        self.training_losses.append(average_loss_vals["_total"].item())
        return

    @torch.no_grad()
    def valid_epoch(self, epoch):
        max_val_iters = self.exp_params["training_slots"].get("val_iters_per_epoch", 1e8)
        self.loss_tracker.reset()
        total_progress_bar = min(len(self.valid_loader), max_val_iters)
        progress_bar = tqdm(enumerate(self.valid_loader), total=total_progress_bar)

        for i, data in progress_bar:
            if i >= max_val_iters:
                break

            _ = self.forward_loss_metric(
                    batch_data=data,
                    training=False
                )

            loss = self.loss_tracker.get_last_losses(total_only=True)
            progress_bar.set_description(f"Epoch {epoch+1} iter {i}: valid loss {loss.item():.5f}. ")

        self.loss_tracker.aggregate()
        average_loss_vals = self.loss_tracker.summary(log=True, get_results=True)

        if not hasattr(self, "rank") or self.rank == 0:
            self.writer.log_full_dictionary(
                    dict=average_loss_vals,
                    step=epoch + 1,
                    plot_name="Valid Loss",
                    dir="Valid Loss",
                )
        self.validation_losses.append(average_loss_vals["_total"].item())

        batch_data = next(iter(self.valid_loader))
        self.visualizations(batch_data=batch_data, epoch=epoch)

        return

    def forward_loss_metric(self, batch_data, training=False, inference_only=False, **kwargs):
        raise NotImplementedError("Base Trainer Module does not implement 'forward_loss_metric'...")

    def visualizations(self, batch_data, epoch, iter_):
        raise NotImplementedError("Base Trainer Module does not implement 'forward_loss_metric'...")
