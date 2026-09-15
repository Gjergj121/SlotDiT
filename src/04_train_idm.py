import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.distributed as dist
from base.predictor_trainer import Trainer as PredictorTrainer
from data.load_data import unwrap_batch_data
from lib.config import Config
from lib.runtime import distributed_setup
from lib.logger import Logger
from models.inverse_dynamics_models import SingleSlotLatentAction, SingleLatentAction


class Trainer(PredictorTrainer):
    def setup_predictor(self):
        action_dim = 6 if self.exp_params["dataset"]["dataset_name"] == "CLIPort" else 2
        common = dict(emb_dim=256, action_dim=action_dim, num_actions=16,
                      num_layers=4, num_heads=8, head_dim=32, mlp_dim=512)
        if self.exp_params["representation"]["type"] == "slots":
            model = self.exp_params["model"]["DINOSAUR"]
            predictor = SingleSlotLatentAction(num_slots=model["num_slots"], slot_dim=model["slot_dim"], **common)
        else:
            model = self.exp_params["model"]["predictor"]["DFoT-VAE"]
            predictor = SingleLatentAction(input_size=model["input_size"], patch_size=model["patch_size"],
                                           in_channels=model["in_channels"], **common)
        super().setup_predictor(predictor=predictor)

    def forward_loss_metric(self, batch_data, training=False, **kwargs):
        videos, _, _, others = unwrap_batch_data(self.exp_params, batch_data)
        videos = videos.to(self.device)
        if self.representation.kind == "videovae":
            latents = torch.cat([self.representation.encode(videos[:, i:i + 1]) for i in range(2)], dim=1)
        else:
            latents = self.representation.encode(videos)
        actions = others.get("actions")
        if actions is None or actions.numel() == 0:
            raise ValueError("IDM training requires action labels")
        targets = actions.to(self.device).float()
        if self.exp_params["dataset"]["dataset_name"] == "LanguageTable-Synthetic":
            targets = targets[:, 0]
        predictions = self.predictor(latents)[:, 0]
        if predictions.shape != targets.shape:
            raise ValueError(f"Action shape mismatch: {predictions.shape}, {targets.shape}")
        loss = F.mse_loss(predictions, targets)
        if training:
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), self.exp_params["training_prediction"]["clipping_max_value"])
            self.optimizer.step()
        return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("-d", "--exp-directory", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume-training", action="store_true")
    args = parser.parse_args()
    if args.resume_training and not args.checkpoint:
        parser.error("--resume-training requires --checkpoint")
    config = json.loads(Path(args.config).read_text())
    name = config["dataset"]["dataset_name"]
    if name not in ("CLIPort", "LanguageTable-Synthetic"):
        parser.error("IDM training supports CLIPort and LanguageTable-Synthetic")
    config["training_prediction"] = dict(num_context=1, num_preds=1, sample_length=2,
                                         num_epochs=1000, scheduler_steps=1000, optimizer="adam", lr=1e-4)
    predictor_name = config["model"]["predictor"]["predictor_name"]
    config["model"]["predictor"][predictor_name].update(use_ema=False, load_ema=False)
    config["dataset"]["sample_rate" if name == "LanguageTable-Synthetic" else "step_size"] = 1
    for key in ("slots_path", "latents_path"):
        config["dataset"].pop(key, None)
    if name == "CLIPort":
        config["dataset"].update(get_actions=True, get_first_and_last_frame=True,
                                  normalize_actions=True, position_only_actions=True)
    rank, world_size, device = distributed_setup()
    destination = Path(args.exp_directory)
    if rank == 0:
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / "experiment_params.json"
        if not target.exists():
            target.write_text(json.dumps(config, indent=2) + "\n")
        elif not args.resume_training:
            raise FileExistsError(f"Experiment already exists: {destination}")
        Logger(str(destination))
    if dist.is_initialized():
        dist.barrier()
    trainer = Trainer(exp_path=str(destination), name_predictor_experiment="", savi_model=None,
                      checkpoint=args.checkpoint, resume_training=args.resume_training,
                      rank=rank, world_size=world_size, device=device)
    try:
        trainer.load_data()
        trainer.setup_predictor()
        trainer.training_loop()
    finally:
        trainer.writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
