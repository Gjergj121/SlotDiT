from pathlib import Path
import torch.distributed as dist
from base.predictor_trainer import Trainer
from lib.logger import Logger
from lib.runtime import training_arguments, distributed_setup


def main():
    args = training_arguments()
    rank, world_size, device = distributed_setup()
    Path(args.exp_directory).mkdir(parents=True, exist_ok=True)
    if rank == 0:
        Logger(args.exp_directory)
    trainer = Trainer(exp_path=args.exp_directory, name_predictor_experiment="", savi_model=None,
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
