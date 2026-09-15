import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_blocks import ConvBlock
import models.timm_encoders as timm_encoders

from models.model_blocks import ResnetBlock, AttnBlock, nonlinearity, Normalize


def get_encoder(encoder_name, in_channels=3, num_blocks=12, **kwargs):
    if encoder_name != "vit_base_patch14_dinov2":
        raise ValueError(f"Unsupported encoder: {encoder_name}")
    backbone = timm_encoders.vit_base_patch14_dinov2(img_size=kwargs.get("img_size", 224))
    return timm_encoders.ViTEncoder(vit_backbone=backbone, num_blocks=num_blocks)

def get_decoder(decoder_name, **kwargs):
    if decoder_name != "MLPPatchDecoder":
        raise ValueError(f"Unsupported decoder: {decoder_name}")
    return MLPPatchDecoder(**kwargs)


class PatchDecoder(nn.Module):
    def __init__(self, num_patches, in_dim):
        super().__init__()
        self.num_patches = num_patches
        self.in_dim = in_dim

        self.pos_embed = nn.Parameter(torch.randn(1, 1, num_patches, in_dim) / (in_dim ** 0.5))
        return

    def forward(self):
        raise NotImplementedError(f"Base Class 'PatchDecoder' does not implement forward pass")

    def broadcast_slots(self, slots):
        broadcasted_slots = slots.unsqueeze(2)
        broadcasted_slots = broadcasted_slots.repeat(1, 1, self.num_patches, 1)
        return broadcasted_slots

    def add_positional_encoding(self, slots):
        B, num_slots, num_patches, slot_dim = slots.shape
        assert num_patches == self.pos_embed.shape[2], f"{num_patches = } different from {self.pos_embed.shape = }"
        assert slot_dim == self.pos_embed.shape[3], f"{slot_dim = } different from {self.pos_embed.shape = }"
        pos_embed = self.pos_embed.repeat(B, num_slots, 1, 1)
        augmented_slots = slots + pos_embed
        return augmented_slots

class MLPPatchDecoder(PatchDecoder):
    def __init__(self, num_patches, in_dim, hidden_dim, out_dim, num_layers=4,
                 initial_layer_norm=False, **kwargs):
        super().__init__(
            num_patches=num_patches,
            in_dim=in_dim
        )
        self.num_patches = num_patches
        self.patch_grid = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.initial_layer_norm = initial_layer_norm
        self.reconstruct_images = kwargs.get("predict_pixels", False)
        self.patch_size = kwargs.get("patch_size", 0)
        self.image_size = kwargs.get("img_size", self.patch_grid[0] * self.patch_size)
        self.num_layers_convPatchDecoder = kwargs.get("num_layers_imgReconsDecoder", num_layers)
        self.imageReconsCNN_name = kwargs.get("imageReconsCNN_name", None)

        self.mlp = self._build_mlp()

        if self.reconstruct_images:
            if self.imageReconsCNN_name is None:
                self.convPatchDecoder = self._build_convPatchDecoder(
                    in_dim= out_dim-1,
                    hidden_dim= self.hidden_dim,
                    num_layers= self.num_layers_convPatchDecoder,
                    patch_size= self.patch_size
                )
            else:
                imageReconsCNN_params = kwargs.get("imageReconsCNN_params", None)
                self.convPatchDecoder = ComplexCNNDecoder(
                    ch= imageReconsCNN_params['ch'],
                    out_ch= 3,
                    ch_mult= imageReconsCNN_params['ch_mult'],
                    num_res_blocks= imageReconsCNN_params['num_res_blocks'],

                    attn_resolutions= [self.patch_grid[0]],
                    dropout= imageReconsCNN_params['dropout'],
                    resamp_with_conv= True,
                    input_resolution= self.patch_grid[0],
                    input_channels= out_dim-1,
                    give_pre_end= False
                )

        return

    def forward(self, slots):
        broadcasted_slots = self.broadcast_slots(slots)
        augmented_slots = self.add_positional_encoding(broadcasted_slots)
        decoded_features = self.mlp(augmented_slots)

        feats, alpha = decoded_features[..., :-1], decoded_features[..., -1:]
        alpha = F.softmax(alpha, dim=1)
        recons_features = torch.sum(feats * alpha, dim=1)

        B, num_slots = alpha.shape[0], alpha.shape[1]
        masks = alpha.reshape(B, num_slots, 1, *self.patch_grid)

        recons_imgs = torch.tensor([])
        if self.reconstruct_images:
            input_decoder = recons_features.permute(0, 2, 1)
            input_decoder = input_decoder.reshape(B, self.out_dim-1, *self.patch_grid)
            recons_imgs = self.convPatchDecoder(input_decoder)

            if recons_imgs.shape[-1] != self.image_size:
                recons_imgs = F.interpolate(recons_imgs, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)

        out = {
            "recons_feats": recons_features,
            "ind_recons_feats": feats,
            "recons_masks": masks,
            "recons_img": recons_imgs
        }
        return recons_features, out

    def _build_mlp(self):
        mlp = []
        if self.initial_layer_norm:
            mlp.append(nn.LayerNorm(self.in_dim))
        for i in range(self.num_layers):
            dim1 = self.hidden_dim if i > 0 else self.in_dim
            dim2 = self.hidden_dim if i < self.num_layers - 1 else self.out_dim
            mlp.append(nn.Linear(dim1, dim2))
            if i < self.num_layers - 1:
                mlp.append(nn.ReLU())
        mlp = nn.Sequential(*mlp)
        return mlp

    def _build_convPatchDecoder(self, in_dim, hidden_dim, num_layers, patch_size):
        modules = []
        current_size = self.patch_grid[0]

        for i in range(num_layers):
            in_channels = in_dim if i == 0 else hidden_dim
            if (i>0) and ((i+1) * 2 < patch_size) and (current_size < self.image_size):
                hidden_dim = hidden_dim // 2
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                batch_norm=True,
            )
            modules.append(block)
            if ((i+1) * 2 < patch_size) and (current_size < self.image_size):
                modules.append(nn.Upsample(scale_factor=2))
                current_size = current_size*2

        final_conv = nn.Conv2d(
                in_channels=hidden_dim,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1
            )
        modules.append(final_conv)

        conv_decoder = nn.Sequential(*modules)
        return conv_decoder


class UpsampleWithConv(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class ComplexCNNDecoder(nn.Module):
    def __init__(self, *, ch, out_ch, ch_mult=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 input_resolution, input_channels, give_pre_end=False, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        self.give_pre_end = give_pre_end

        block_in = ch*ch_mult[self.num_resolutions-1]

        curr_res = input_resolution

        self.conv_in = torch.nn.Conv2d(input_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = UpsampleWithConv(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        self.last_z_shape = z.shape

        temb = None

        h = self.conv_in(z)

        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
