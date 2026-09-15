import os


import random

import datetime

import numpy as np

import torch

from torch.utils.tensorboard import SummaryWriter


from collections import OrderedDict


from lib.logger import log_function, print_

from CONFIG import CONFIG

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def set_random_seed(random_seed=None):
    if(random_seed is None):
        random_seed = CONFIG["random_seed"]
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    return

@log_function
def create_directory(dir_path, dir_name=None):
    if(dir_name is not None):
        dir_path = os.path.join(dir_path, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    return

def timestamp():
    timestamp = str(datetime.datetime.now()).split('.')[0].replace(' ', '_').replace(':', '-')
    return timestamp

@log_function
def log_architecture(model, exp_path, fname="model_architecture.txt"):
    assert fname[-4:] == ".txt", "ERROR! 'fname' must be a .txt file"
    savepath = os.path.join(exp_path, fname)

    with open(savepath, "w") as f:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total Params: {num_params}")

    for i, layer in enumerate(model.children()):
        if(isinstance(layer, torch.nn.Module)):
            log_module(module=layer, exp_path=exp_path, fname=fname)
    return

def log_module(module, exp_path, fname="model_architecture.txt", append=True):
    assert fname[-4:] == ".txt", "ERROR! 'fname' must be a .txt file"
    savepath = os.path.join(exp_path, fname)

    if (append is False):
        with open(savepath, "w") as f:
            f.write("")
    else:
        with open(savepath, "a") as f:
            f.write("\n\n")

    with open(savepath, "a") as f:
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        f.write(f"Params: {num_params}")
        f.write("\n")
        f.write(str(module))
    return

def get_from_dict(my_dict, key, default="no-default", msg=None, pop=False):
    if key not in my_dict.keys() and default == "no-default":
        if msg is None:
            msg = f"Required {key = } not found in dictionary with keys {dict.keys()}"
        raise KeyError(msg)
    out = my_dict.pop(key, default) if pop else my_dict.get(key, default)
    return out

class TensorboardWriter:
    def close(self):
        self.writer.close()

    def __init__(self, logdir):
        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        return

    def add_scalar(self, name, val, step):
        self.writer.add_scalar(name, val, step)
        return

    def add_scalars(self, plot_name, val_names, vals, step):
        val_dict = {val_name: val for (val_name, val) in zip(val_names, vals)}
        self.writer.add_scalars(plot_name, val_dict, step)
        return

    def add_image(self, fig_name, img_grid, step):
        self.writer.add_image(fig_name, img_grid, global_step=step)
        return

    def add_images(self, fig_name, img_grid, step):
        self.writer.add_images(fig_name, img_grid, global_step=step)
        return

    def add_figure(self, tag, figure, step):
        self.writer.add_figure(tag=tag, figure=figure, global_step=step)
        return

    def add_graph(self, model, input):
        self.writer.add_graph(model, input_to_model=input)
        return

    def log_full_dictionary(self, dict, step, plot_name="Losses", dir=None):
        if dir is not None:
            dict = {f"{dir}/{key}": val for key, val in dict.items()}
        else:
            dict = {key: val for key, val in dict.items()}

        for key, val in dict.items():
            self.add_scalar(name=key, val=val, step=step)

        plot_name = f"{dir}/{plot_name}" if dir is not None else key
        self.add_scalars(plot_name=plot_name, val_names=dict.keys(), vals=dict.values(), step=step)
        return
