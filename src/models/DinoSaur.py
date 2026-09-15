import math
import torch
import torch.nn as nn

from models import SlotAttention, get_encoder, get_decoder, get_initalizer
from models.transition_models import get_transition_module
from models.model_utils import init_xavier_
from lib.logger import print_

class DinoSaur(nn.Module):
    SUPPORTED_ENCODERS = ["vit_base_patch14_dinov2"]
    SUPPORTED_DECODERS = ["MLPPatchDecoder"]

    def __init__(self, num_slots, slot_dim, num_iterations=3, in_channels=3, mlp_hidden=128,
                 mlp_encoder_dim=128, initializer=None, encoder_name=None, encoder_num_blocks=None,
                 decoder_name=None, decoder_params=None, transition_module_params=None, **kwargs):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations_first = kwargs.get("num_iterations_first", num_iterations)
        self.num_iterations = num_iterations
        self.in_channels = in_channels
        self.mlp_hidden = mlp_hidden
        self.mlp_encoder_dim = mlp_encoder_dim
        self.use_linear_feat_proj = kwargs.get("linear_feat_proj", False)
        self.encoder_name = encoder_name
        self.img_size = kwargs.get("img_size", decoder_params['patch_size']*int(decoder_params['num_patches'] ** 0.5))
        self.decoder_name = decoder_name

        print_("Initializer:")
        print_(f"  --> mode={initializer}")
        print_(f"  --> slot_dim={slot_dim}")
        print_(f"  --> num_slots={num_slots}")
        self.initializer = get_initalizer(
                mode=initializer,
                slot_dim=slot_dim,
                num_slots=num_slots
            )

        if encoder_name not in DinoSaur.SUPPORTED_ENCODERS:
            raise ValueError(f"{encoder_name} not supported in DinoSaur")
        print_("Encoder:")
        print_(f"  --> Encoder_name={encoder_name}")
        print_(f"  --> in_channels={in_channels}")
        self.encoder = get_encoder(
                encoder_name=encoder_name,
                in_channels=in_channels,
                num_blocks=encoder_num_blocks,
                img_size=self.img_size
            )

        if self.use_linear_feat_proj:
            linear_feat_proj = nn.Sequential(
                nn.LayerNorm(mlp_encoder_dim),
                nn.Linear(mlp_encoder_dim, mlp_encoder_dim),
                nn.ReLU(),
                nn.Linear(mlp_encoder_dim, slot_dim),
            )
            sa_in_dim = slot_dim
        else:
            linear_feat_proj = nn.Identity()
            sa_in_dim = mlp_encoder_dim
        self.linear_feat_proj = linear_feat_proj

        self.predictor = get_transition_module(
            model_name=transition_module_params.pop('transition_module_name'),
            **transition_module_params
        )

        if decoder_name not in DinoSaur.SUPPORTED_DECODERS:
            raise ValueError(f"{decoder_name} not supported in DinoSaur")

        print_("Decoder:")
        print_(f"  --> Decoder_name={decoder_name}")
        decoder_params['img_size']= self.img_size
        decoder_params['feat_dim']= self.mlp_encoder_dim
        self.decoder = get_decoder(decoder_name, **decoder_params)

        self.slot_attention = SlotAttention(
            dim_feats=sa_in_dim,
            dim_slots=slot_dim,
            num_slots=num_slots,
            num_iters_first=self.num_iterations_first,
            num_iters=num_iterations,
            mlp_hidden=mlp_hidden,
        )

        self._init_model()
        self.reset()
        return

    @torch.no_grad()
    def _init_model(self):
        init_xavier_(self.linear_feat_proj)
        init_xavier_(self.predictor)
        init_xavier_(self.slot_attention)
        init_xavier_(self.decoder)

        torch.nn.init.zeros_(self.slot_attention.gru.bias_ih)
        torch.nn.init.zeros_(self.slot_attention.gru.bias_hh)
        torch.nn.init.orthogonal_(self.slot_attention.gru.weight_hh)
        if hasattr(self.slot_attention, "slots_mu"):
            limit = math.sqrt(6.0 / (1 + self.slot_attention.dim_slots))
            torch.nn.init.uniform_(self.slot_attention.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.slot_attention.slots_sigma, -limit, limit)
        return

    def forward(self, input, num_imgs=10, **kwargs):
        B, T, C, H, W = input.shape

        outs = []
        self.reset()

        predicted_slots = self.initializer(batch_size=B, **kwargs)

        input_reshaped = input.reshape(B*T, C, H, W)
        with torch.no_grad():
            img_feats_seq = self.encoder(input_reshaped)

        proj_img_feats = self.linear_feat_proj(img_feats_seq)
        proj_img_feats_reshaped = proj_img_feats.reshape(B, T, *proj_img_feats.shape[1:])

        for t in range(num_imgs):
            cur_outs = {}
            img_feats = proj_img_feats_reshaped[:, t]

            slots = self.apply_attention(
                x=img_feats,
                predicted_slots=predicted_slots,
                step=t
            )

            predicted_slots, transition_module_stats = self.apply_transition_module(slots)
            cur_outs["slot_history"] = slots
            cur_outs = {**cur_outs, **transition_module_stats}
            outs.append(cur_outs)

        outs = {key: torch.stack([d[key] for d in outs], dim=1) for key in outs[0].keys()}
        outs["encoded_img_feats"] = img_feats_seq.clone().reshape(B, T, img_feats_seq.shape[1], img_feats_seq.shape[2])

        slot_history = outs['slot_history'].clone().reshape(B*T, self.num_slots, self.slot_dim)
        feats_recons, others_decoder = self.decode(slot_history, teacher_features=img_feats_seq)

        outs["recons_feats"] = feats_recons.reshape(B, T, *feats_recons.shape[1:])
        others_decoder = {
            key: (value.reshape(B, T, *value.shape[1:]) if value.numel() > 0 else value)
            for key, value in others_decoder.items()
        }
        outs = {**outs, **others_decoder}

        return outs


    def apply_transition_module(self, slots):
        predicted_slots = self.predictor(slots)
        if isinstance(predicted_slots, tuple):
            predicted_slots, others = predicted_slots
        else:
            others = {}
        return predicted_slots, others

    def apply_attention(self, x, predicted_slots=None, step=0):
        slots = self.slot_attention(
            inputs=x,
            slots=predicted_slots,
            step=step
        )
        return slots

    def encode(self, images):
        if images.dim() == 5:
            B, T, C, H, W = images.shape
        elif images.dim() == 4:
            images = images.unsqueeze(1)
            B, T, C, H, W = images.shape
        else:
            images = images.unsqueeze(0).unsqueeze(0)
            B, T, C, H, W = images.shape

        outs = []
        self.reset()

        predicted_slots = self.initializer(batch_size=B)

        images = images.reshape(B*T, C, H, W)

        with torch.no_grad():
            img_feats_seq = self.encoder(images)

        proj_img_feats = self.linear_feat_proj(img_feats_seq)
        proj_img_feats_reshaped = proj_img_feats.reshape(B, T, *proj_img_feats.shape[1:])

        for t in range(T):
            cur_outs = {}
            img_feats = proj_img_feats_reshaped[:, t]

            slots = self.apply_attention(
                x=img_feats,
                predicted_slots=predicted_slots,
                step=t
            )

            predicted_slots, transition_module_stats = self.apply_transition_module(slots)
            cur_outs["slot_history"] = slots
            cur_outs = {**cur_outs, **transition_module_stats}
            outs.append(cur_outs)

        outs = {key: torch.stack([d[key] for d in outs], dim=1) for key in outs[0].keys()}

        return outs["slot_history"]

    def decode(self, slots, teacher_features=None):
        if "Transformer" in self.decoder_name:
            teacher_features = teacher_features[:, :-1]
            feats_recons, others_decoder = self.decoder(x=teacher_features, slots=slots)
        else:
            feats_recons, others_decoder = self.decoder(slots)
        return feats_recons, others_decoder

    def reset(self):
        if hasattr(self.predictor, "reset"):
            self.predictor.reset()
        return
