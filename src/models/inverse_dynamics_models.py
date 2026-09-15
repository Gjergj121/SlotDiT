import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.attention import TransformerEncoderBlock, MLP
from models.model_blocks import PositionalEncoding, TemporalPositionalEncoding

class BaseSlotLatentAction(nn.Module):
    def __init__(self, slot_dim, emb_dim, action_dim, num_actions,
                 use_ema_vq=False,**kwargs):
        super().__init__()
        self.slot_dim = slot_dim
        self.emb_dim = emb_dim
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.use_ema_vq = use_ema_vq

        self._setup_slot_encoder()
        return

    def sample(self, mean, var, eps=1e-6):
        noise = torch.randn(mean.size(), dtype=torch.float32).to(mean.device)
        z = noise * torch.sqrt(var + eps) + mean
        return z

    def comput_action_dist(self, tokens):
        mean_token = self.mean_fc(tokens)
        var_token = torch.abs(self.variance_fc(tokens))
        action_dir_mean = mean_token[:, 1:] - mean_token[:, :-1]
        action_dir_var = var_token[:, 1:] + var_token[:, :-1]
        return action_dir_mean, action_dir_var

    def compute_actions(self, slots):
        return self(slots)

    def get_action(self, action_idx=None, shape=None):
        assert shape is not None, f"A shape argument must be specified..."
        device = self.mean_fc.weight.device
        if action_idx is None:
            action_idx = torch.randint(
                    low=0,
                    high=self.quantizer.num_embs,
                    size=shape,
                    device=device
                )
        else:
            action_idx = torch.tensor(action_idx, device=device).expand(shape)
        action_protos = self.quantizer.get_codebook_entry(action_idx)
        return action_protos, action_idx

    def decompose_action_latent(self, action_latent):
        action_protos, _, _ = self.quantizer(action_latent)
        action_variability, _ = self.quantizer.get_variability(
                z=action_latent,
                action_embs=action_protos
            )
        return action_protos, action_variability

class SingleSlotLatentAction(BaseSlotLatentAction):
    def __init__(self, num_slots, slot_dim, emb_dim, action_dim, num_actions,
                 num_layers, num_heads, head_dim, mlp_dim,
                 use_ema_vq=False, **kwargs):
        super().__init__(
                slot_dim=slot_dim,
                emb_dim=emb_dim,
                action_dim=action_dim,
                num_actions=num_actions,
                use_ema_vq=use_ema_vq
            )

        self.cond_dim = kwargs.get("cond_dim", 0)
        if self.cond_dim > 0:
            self.cond_encoder = nn.Linear(self.cond_dim, emb_dim)

        self.act_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.transformer = nn.Sequential(*[
                TransformerEncoderBlock(
                    embed_dim=emb_dim,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_dim,
                    project_out=True
                )
            for _ in range(num_layers)])

        self.mlp = MLP(
            in_dim=emb_dim,
            hidden_dim=mlp_dim,
            out_dim=action_dim
        )

        self.pe = TemporalPositionalEncoding(
                d_model=self.emb_dim,
                max_len=num_slots + 1,
                mode="learned"
            )
        return

    def _setup_slot_encoder(self):
        self.slot_encoder = nn.Sequential(
                nn.LayerNorm(self.slot_dim),
                nn.Linear(self.slot_dim, self.emb_dim)
            )
        return

    def forward(self, slots, cond=None):
        if len(slots.shape) != 4:
            raise ValueError(f"{slots.shape = } must be (B, N, num_slots, slot_dim)")
        B, N, num_slots, _ = slots.shape

        slot_embs = self.slot_encoder(slots.flatten(0, 2)).reshape(B, N, num_slots, -1)
        slot_embs = self.pe(x=slot_embs, batch_size=B, num_slots=num_slots)
        slots_embs = slot_embs.reshape(B, N*num_slots, -1)
        act_tokens = self.act_token.repeat(B, N-1, 1)
        all_tokens = torch.cat([act_tokens, slots_embs], dim=1)
        if self.cond_dim > 0 and cond is not None:
            cond_token = self.cond_encoder(cond.to(all_tokens.dtype)).unsqueeze(1)
            all_tokens = torch.cat([all_tokens, cond_token], dim=1)

        output_token = self.transformer(all_tokens)[:, :N-1, :]

        output_action = self.mlp(output_token)

        return output_action

class SingleLatentAction(BaseSlotLatentAction):
    def __init__(self, input_size, patch_size, in_channels, emb_dim, action_dim,
                 num_actions, num_layers, num_heads, head_dim, mlp_dim,
                 use_ema_vq=False, max_frames=2, **kwargs):
        super().__init__(
                slot_dim=in_channels,
                emb_dim=emb_dim,
                action_dim=action_dim,
                num_actions=num_actions,
                use_ema_vq=use_ema_vq
            )

        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = (input_size // patch_size) ** 2

        from timm.models.vision_transformer import PatchEmbed
        self.x_embedder = PatchEmbed(
            input_size,
            patch_size,
            in_channels,
            emb_dim,
            bias=True
        )

        self.act_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.transformer = nn.Sequential(*[
                TransformerEncoderBlock(
                    embed_dim=emb_dim,
                    head_dim=head_dim,
                    num_heads=num_heads,
                    mlp_size=mlp_dim,
                    project_out=True
                )
            for _ in range(num_layers)])

        self.mlp = MLP(
            in_dim=emb_dim,
            hidden_dim=mlp_dim,
            out_dim=action_dim
        )

        from models.model_blocks import LearnedPositionalEncoding1D
        self.pe = LearnedPositionalEncoding1D(
            max_len=self.num_patches*max_frames,
            token_dim=emb_dim,
            dropout=0.0
        )
        return

    def _setup_slot_encoder(self):
        pass

    def forward(self, latents):
        if len(latents.shape) != 5:
            raise ValueError(
                f"{latents.shape = } must be (B, N, num_channels, h, w)"
            )
        B, N, C, H, W = latents.shape

        latents_flat = rearrange(latents, "b t c h w -> (b t) c h w")

        patch_embs = self.x_embedder(latents_flat)

        patch_embs = rearrange(patch_embs, "(b t) p c -> b t p c", b=B)

        patch_embs = patch_embs.reshape(B, N * self.num_patches, -1)

        patch_embs = self.pe(patch_embs)

        act_tokens = self.act_token.repeat(B, N-1, 1)

        all_tokens = torch.cat([act_tokens, patch_embs], dim=1)

        output_tokens = self.transformer(all_tokens)

        output_token = output_tokens[:, :N-1, :]

        output_action = self.mlp(output_token)

        return output_action
