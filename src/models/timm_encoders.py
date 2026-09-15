from functools import partial
import torch
import torch.nn as nn

import timm

from models.model_utils import freeze_params

class ViTEncoder(nn.Module):
    def __init__(self, vit_backbone, num_blocks=None, backbone_name=None):
        assert hasattr(vit_backbone, "blocks"), f"ViT backbone must have attribute 'blocks'"
        if num_blocks is not None:
            assert len(vit_backbone.blocks) >= num_blocks and num_blocks > 0, \
            f"{num_blocks =} must be in 0 < num_blocks <= {len(vit_backbone.blocks) = }"
        super().__init__()
        self.vit_backbone = vit_backbone
        self.num_blocks = num_blocks
        self.backbone_name = backbone_name

        if num_blocks is not None:
            self.vit_backbone.blocks = self.vit_backbone.blocks[:num_blocks]
        freeze_params(self)

        self.mean = torch.tensor(self.vit_backbone.default_cfg["mean"]).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(self.vit_backbone.default_cfg["mean"]).view(1, 1, 3, 1, 1)
        return

    def forward(self, x):
        x = self.normalize_images(x)
        x = self.vit_backbone.patch_embed(x)
        if "dinov3" != self.backbone_name:
            x = self.vit_backbone._pos_embed(x)
            x = self.vit_backbone.patch_drop(x)
        else:
            x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = self.vit_backbone.norm_pre(x)
        x = self.vit_backbone.blocks(x)
        if "dinov3" != self.backbone_name:
            x = x[:, 1:]
        return x

    @torch.no_grad()
    def _get_num_patches(self):
        dummy = torch.randn(1, 3, 224, 224)
        out = self.forward(dummy)
        num_patches = out.shape[1]
        return num_patches

    def normalize_images(self, img):
        if self.mean.device != img.device:
            self.mean = self.mean.to(img.device)
            self.std = self.std.to(img.device)
        if len(img.shape) == 4:
            mean, std = self.mean[0], self.std[0]
        elif len(img.shape) == 5:
            mean, std = self.mean, self.std
        else:
            raise ValueError(f"Weird input shape: {img.shape = }. It should be either 4- or 5-dimensional")
        norm_img = (img - mean) / std
        return norm_img


def vit_base_patch14_dinov2(pretrained=True, **kwargs):
    model_kwargs = dict(
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_classes=0,
        **kwargs,
    )
    model = timm.create_model(
        "vit_base_patch14_dinov2.lvd142m",
        pretrained=pretrained,
        **model_kwargs
    )
    return model
