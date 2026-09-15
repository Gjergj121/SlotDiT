import os
import traceback
import torch

from lib.logger import print_, log_function
from lib.schedulers import LRWarmUp, ExponentialLRSchedule
from lib.utils import create_directory
import models
import models.Predictors as predictors
from CONFIG import MODELS, PREDICTORS
from models.DiT import DFoT, DFoT_VAE

def setup_model(model_params):
    from copy import deepcopy
    if model_params["model_name"] != "DINOSAUR":
        raise ValueError("Only DINOSAUR is included in this release")
    return models.DinoSaur(**deepcopy(model_params["DINOSAUR"]))


def setup_predictor(exp_params, **kwargs):
    model_params = exp_params["model"]
    predictor_params = model_params["predictor"]
    predictor_name = predictor_params["predictor_name"]
    cur_predictor_params = predictor_params[predictor_name]
    if predictor_name.lower() == 'dfot':
        cur_model_params = model_params[model_params['model_name']]
        predictor = DFoT(
            num_slots=cur_model_params['num_slots'],
            input_dim=cur_model_params['slot_dim'],
            hidden_size=cur_predictor_params['model_hidden_size'],
            depth=cur_predictor_params['num_blocks'],
            num_heads=cur_predictor_params['num_heads'],
            mlp_ratio=cur_predictor_params['mlp_ratio'],
            text_dropout_prob=cur_predictor_params['text_dropout_prob'],
            learn_sigma=cur_predictor_params['learn_sigma'],
            buffer_size=cur_predictor_params['buffer_size'],
            caption_channels=cur_predictor_params['text_encoder_params']['hidden_size'],
            model_max_length=cur_predictor_params['text_encoder_params']['token_max_length'],
            layer_norm_ca=cur_predictor_params['layer_norm_ca'],
            pooling=cur_predictor_params['pooling'],
            zero_init=cur_predictor_params.get('zero_init', True),
            use_rope=cur_predictor_params.get('use_rope', False),
            adaln_temp=cur_predictor_params.get('adaln_temp', False),
            adaln_cross_attention=cur_predictor_params.get('adaln_cross_attention', False),
            use_temp_attn_layers=cur_predictor_params.get('use_temp_attn_layers', True),
            use_qk_norm=cur_predictor_params.get('use_qk_norm', False),
            preserve_slot_equivariance=cur_predictor_params.get('preserve_slot_equivariance', True),
            drop_path=cur_predictor_params.get('drop_path', 0.0),
        )
    elif predictor_name.lower() == 'dfot-vae':
        predictor = DFoT_VAE(
            input_size=cur_predictor_params['input_size'],
            patch_size=cur_predictor_params['patch_size'],
            in_channels=cur_predictor_params['in_channels'],
            hidden_size=cur_predictor_params['model_hidden_size'],
            depth=cur_predictor_params['num_blocks'],
            num_heads=cur_predictor_params['num_heads'],
            mlp_ratio=cur_predictor_params['mlp_ratio'],
            text_dropout_prob=cur_predictor_params['text_dropout_prob'],
            learn_sigma=cur_predictor_params['learn_sigma'],
            buffer_size=cur_predictor_params['buffer_size'],
            caption_channels=cur_predictor_params['text_encoder_params']['hidden_size'],
            model_max_length=cur_predictor_params['text_encoder_params']['token_max_length'],
            layer_norm_ca=cur_predictor_params['layer_norm_ca'],
            zero_init=cur_predictor_params.get('zero_init', True),
            use_rope=cur_predictor_params.get('use_rope', False),
            adaln_temp=cur_predictor_params.get('adaln_temp', False),
            adaln_cross_attention=cur_predictor_params.get('adaln_cross_attention', False),
            use_temp_attn_layers=cur_predictor_params.get('use_temp_attn_layers', True),
            use_qk_norm=cur_predictor_params.get('use_qk_norm', False),
            use_vq=cur_predictor_params.get('use_vq', False),
            drop_path=cur_predictor_params.get('drop_path', 0.0),
        )
    else:
        raise ValueError(f"Unsupported predictor: {predictor_name}")
    return predictors.PredictorWrapper(exp_params=exp_params, predictor=predictor)


@log_function
def save_checkpoint(model, optimizer, scheduler, lr_warmup, epoch, exp_path,
                    finished=False, savedir="models", savename=None):
    if(savename is not None):
        checkpoint_name = savename
    elif(savename is None and finished is True):
        checkpoint_name = "checkpoint_epoch_final.pth"
    else:
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth"

    create_directory(exp_path, savedir)
    savepath = os.path.join(exp_path, savedir, checkpoint_name)

    scheduler_data = "" if scheduler is None else scheduler.state_dict()
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "scheduler_state_dict": scheduler_data,
            "lr_warmup": lr_warmup
        }, savepath)

    return

@log_function
def load_checkpoint(checkpoint_path, model, only_model=False, map_cpu=False, **kwargs):
    if checkpoint_path is None:
        return model
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist ...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    first_key_model = list(model.state_dict().keys())[0]
    first_key_checkpoint = list(checkpoint['model_state_dict'].keys())[0]
    if first_key_model.startswith("predictor") and not first_key_checkpoint.startswith("predictor"):
        checkpoint['model_state_dict'] = {
                f"predictor.{key}": val for key, val in checkpoint['model_state_dict'].items()
            }

    model.load_state_dict(checkpoint['model_state_dict'])

    if(only_model):
        return model

    optimizer, scheduler, lr_warmup = kwargs["optimizer"], kwargs["scheduler"], kwargs["lr_warmup"]
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if "lr_warmup" in checkpoint:
        lr_warmup.load_state_dict(checkpoint['lr_warmup'])
    epoch = checkpoint["epoch"] + 1

    return model, optimizer, scheduler, lr_warmup, epoch

@log_function
def setup_optimizer(exp_params, model, section="training_slots"):
    lr = exp_params[section]["lr"]
    lr_factor = exp_params[section]["lr_factor"]
    patience = exp_params[section]["patience"]
    momentum = exp_params[section]["momentum"]
    optimizer = exp_params[section]["optimizer"]
    nesterov = exp_params[section]["nesterov"]
    scheduler = exp_params[section]["scheduler"]
    scheduler_steps = exp_params[section].get("scheduler_steps", 1e6)

    if(optimizer == "adam"):
        print_("Setting up Adam optimizer:")
        print_(f"    LR: {lr}")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=exp_params[section]["weight_decay"])
    elif optimizer == "sgd":
        print_("Setting up SGD optimizer:")
        print_(f"    LR: {lr}")
        print_(f"    Momentum: {momentum}")
        print_(f"    Nesterov: {nesterov}")
        print_("    Weight Decay: 0.0005")
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                    nesterov=nesterov, weight_decay=0.0005)

    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    if (scheduler == "constant"):
        print_("Setting up Constant LR-Scheduler:")
        print_(f"   Factor:   {lr_factor}")
        print_(f"   total_iters: {scheduler_steps}")
        scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer=optimizer,
                factor=1,
                total_iters=scheduler_steps
            )
    elif(scheduler == "plateau"):
        print_("Setting up Plateau LR-Scheduler:")
        print_(f"   Patience: {patience}")
        print_(f"   Factor:   {lr_factor}")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                patience=patience,
                factor=lr_factor,
                min_lr=1e-8,
                mode="min",
                verbose=True
            )
    elif(scheduler == "step"):
        print_("Setting up Step LR-Scheduler")
        print_(f"   Step Size: {patience}")
        print_(f"   Factor:    {lr_factor}")
        scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer,
                gamma=lr_factor,
                step_size=patience
            )

    elif(scheduler == "multi_step"):
        print_("Setting up MultiStepLR LR-Scheduler")
        print_(f"   Milestones: {patience}")
        print_(f"   Factor:    {lr_factor}")
        if not isinstance(patience, list):
            raise ValueError(f"Milestones ({patience}) must be a list of increasing integers...")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                gamma=lr_factor,
                milestones=patience
            )

    elif(scheduler == "exponential"):
        print_("Setting up Exponential LR-Scheduler")
        print_(f"   Init LR: {lr}")
        print_(f"   Factor:  {lr_factor}")
        print_(f"   Steps:   {scheduler_steps}")
        scheduler = ExponentialLRSchedule(
                optimizer=optimizer,
                init_lr=lr,
                gamma=lr_factor,
                total_steps=scheduler_steps
            )
    elif(scheduler == "cosine_annealing"):
        print_("Setting up Cosine Annealing LR-Scheduler")
        print_(f"   Init LR: {lr}")
        print_(f"   Factor:  {lr_factor}")
        print_(f"   T_max:   {scheduler_steps}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=scheduler_steps,
                eta_min=1e-7
            )
    else:
        print_("Not using any LR-Scheduler")
        scheduler = None

    lr_warmup = setup_lr_warmup(params=exp_params[section])

    return optimizer, scheduler, lr_warmup

@log_function
def setup_lr_warmup(params):
    use_warmup = params["lr_warmup"]
    lr = params["lr"]
    if(use_warmup):
        warmup_steps = params["warmup_steps"]
        warmup_epochs = params["warmup_epochs"]
        lr_warmup = LRWarmUp(init_lr=lr, warmup_steps=warmup_steps, max_epochs=warmup_epochs)
        print_("Setting up learning rate warmup:")
        print_(f"  Target LR:     {lr}")
        print_(f"  Warmup Steps:  {warmup_steps}")
        print_(f"  Warmup Epochs: {warmup_epochs}")
    else:
        lr_warmup = LRWarmUp(init_lr=lr, warmup_steps=-1, max_epochs=-1)
        print_("Not using learning rate warmup...")
    return lr_warmup

def emergency_save(function):
    def try_call_except(self, *args, **kwargs):
        try:
            return function(self, *args, **kwargs)
        except (Exception, KeyboardInterrupt):
            if getattr(self, "rank", 0) == 0 and hasattr(self, "optimizer"):
                try:
                    self.wrapper_save_checkpoint(epoch=self.epoch, savename=f"emergency_checkpoint_epoch_{self.epoch}.pth")
                except Exception:
                    print_(traceback.format_exc())
            raise
    return try_call_except
