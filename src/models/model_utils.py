import numpy as np
import torch
import torch.nn as nn

def build_grid(resolution, vmin=-1., vmax=1., device=None):
    ranges = [np.linspace(vmin, vmax, num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    torch_grid = torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)
    return torch_grid

def freeze_params(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

@torch.no_grad()
def init_xavier_(model: nn.Module):
    for name, tensor in model.named_parameters():
        if name.endswith(".bias") or tensor.dtype == torch.bool:
            tensor.zero_()
        elif len(tensor.shape) <= 1:
            pass
        else:
            torch.nn.init.xavier_uniform_(tensor)
