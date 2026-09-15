import os

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG = {
    "random_seed": 14,
    "epsilon_min": 1e-16,
    "epsilon_max": 1e16,
    "num_workers": int(os.environ.get("SLOTDIT_NUM_WORKERS", "8")),
    "paths": {
        "data_path": str(ROOT / "datasets"),
        "experiments_path": str(ROOT / "experiments"),
        "configs_path": str(ROOT / "src" / "configs"),
    },
}

DATASETS = ['CLIPort', 'LanguageTable-Synthetic', 'LanguageTable-Real', 'BridgeV2']

MODELS = ['DINOSAUR']

PREDICTORS = ['DFoT', 'DFoT-VAE']

METRICS = ['mse', 'psnr', 'ssim', 'lpips', 'fvd']

LOSSES = ['mse', 'pred_feature_mse']

INITIALIZERS = ['Random', 'Learned', 'LearnedRandom', 'CoM']

# NOTE: some config placeholders.
DEFAULTS = {
    "training_slots": {
        "num_epochs": 1000,
        "save_frequency": 10,
        "log_frequency": 25,
        "batch_size": 8,
        "lr": 0.0001,
        "optimizer": "adam",
        "momentum": 0,
        "weight_decay": 0,
        "nesterov": False,
        "scheduler": "cosine_annealing",
        "lr_factor": 0.8,
        "patience": 10,
        "scheduler_steps": 1000,
        "lr_warmup": False,
        "warmup_steps": 2000,
        "warmup_epochs": 2,
        "gradient_clipping": True,
        "clipping_max_value": 0.05
    },
    "training_prediction": {
        "num_epochs": 1500,
        "save_frequency": 10,
        "log_frequency": 25,
        "batch_size": 8,
        "lr": 0.0002,
        "optimizer": "adam",
        "momentum": 0,
        "weight_decay": 0,
        "nesterov": False,
        "scheduler": "cosine_annealing",
        "lr_factor": 0.8,
        "patience": 10,
        "scheduler_steps": 1500,
        "lr_warmup": False,
        "warmup_steps": 2000,
        "warmup_epochs": 2,
        "gradient_clipping": True,
        "clipping_max_value": 0.05,
        "num_context": 1,
        "num_preds": 9,
        "train_iters_per_epoch": 10000000000.0,
        "save_frequency_iters": 1000,
        "sample_length": 10
    },
    "evaluation": {
        "num_context": 1,
        "num_preds": 29,
        "history_context": 4,
        "cfg_scale": 1.5,
        "num_sampling_steps": 50
    },
    "loss": [
        {
            "type": "mse",
            "weight": 1
        },
        {
            "type": "pred_feature_mse",
            "weight": 1
        }
    ]
}
