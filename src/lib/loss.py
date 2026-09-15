import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.logger import log_info
from CONFIG import LOSSES


def _mse_with_optional_mask(preds, targets, action_masks=None):
    if preds.shape != targets.shape:
        raise ValueError(f"preds and targets must have the same shape, got {preds.shape} and {targets.shape}")

    if action_masks is None:
        return F.mse_loss(preds, targets)

    mask = action_masks.float()
    while mask.dim() < preds.dim():
        mask = mask.unsqueeze(-1)

    squared_error = (preds - targets) ** 2
    masked_squared_error = squared_error * mask
    valid_elems = mask.expand_as(squared_error).sum().clamp(min=1.0)
    return masked_squared_error.sum() / valid_elems

class LossTracker:
    def __init__(self, loss_params, **kwargs):
        assert isinstance(loss_params, list), f"Loss_params must be a list, not {type(loss_params)}"
        for loss in loss_params:
            if loss["type"] not in LOSSES:
                raise NotImplementedError(f"Loss {loss['type']} not implemented. Use one of {LOSSES}")

        self.device = kwargs.get("device", 'cpu')

        self.loss_computers = {}
        for loss in loss_params:
            loss['device'] = self.device
            loss_type, loss_weight = loss["type"], loss["weight"]
            self.loss_computers[loss_type] = {}
            self.loss_computers[loss_type]["metric"] = get_loss(loss_type, **loss)
            self.loss_computers[loss_type]["weight"] = loss_weight
        self.reset()
        return

    def reset(self):
        self.loss_values = {loss: [] for loss in self.loss_computers.keys()}
        self.loss_values["_total"] = []
        return

    def __call__(self, **kwargs):
        self.accumulate(**kwargs)

    def accumulate(self, **kwargs):
        total_loss = 0
        for loss in self.loss_computers:
            loss_val = self.loss_computers[loss]["metric"](**kwargs)
            self.loss_values[loss].append(loss_val.cpu())
            total_loss = total_loss + loss_val * self.loss_computers[loss]["weight"]
        self.loss_values["_total"].append(total_loss.cpu())
        return

    def aggregate(self):
        self.loss_values["mean_loss"] = {}
        for loss in self.loss_computers:
            self.loss_values["mean_loss"][loss] = torch.stack(self.loss_values[loss]).mean()
        self.loss_values["mean_loss"]["_total"] = torch.stack(self.loss_values["_total"]).mean()
        return

    def get_last_losses(self, total_only=False):
        if total_only:
            last_losses = self.loss_values["_total"][-1]
        else:
            last_losses = {loss: loss_vals[-1] for loss, loss_vals in self.loss_values.items()}
        return last_losses

    def summary(self, log=True, get_results=True):
        if log:
            log_info("LOSS VALUES:")
            log_info("--------")
            for loss, loss_value in self.loss_values["mean_loss"].items():
                log_info(f"  {loss}:  {round(loss_value.item(), 5)}")

        return_val = self.loss_values["mean_loss"] if get_results else None
        return return_val

def get_loss(loss_type="mse", **kwargs):
    classes = {"mse": MSELoss, "pred_feature_mse": PredFeatureMSELoss}
    if loss_type not in classes:
        raise ValueError(f"Unsupported loss: {loss_type}")
    return classes[loss_type]()

class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, **kwargs):
        if "pred_imgs" not in kwargs:
            raise ValueError("'pred_imgs' must be given to LossTracker to compute 'MSELoss'")
        if "target_imgs" not in kwargs:
            raise ValueError("'target_imgs' must be given to LossTracker to compute 'MSELoss'")
        preds, targets = kwargs.get("pred_imgs"), kwargs.get("target_imgs")
        action_masks = kwargs.get("action_masks", None)
        loss = _mse_with_optional_mask(preds, targets, action_masks=action_masks)
        return loss


class PredFeatureMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, **kwargs):
        if "preds_feats" not in kwargs:
            raise ValueError("'pred' must be given to LossTracker to compute 'PredSlotMSELoss'")
        if "targets_feats" not in kwargs:
            raise ValueError("'target_slots' must be given to LossTracker to compute 'PredSlotMSELoss'")
        preds, targets = kwargs.get("preds_feats"), kwargs.get("targets_feats")
        loss = self.mse(preds, targets)
        return loss
