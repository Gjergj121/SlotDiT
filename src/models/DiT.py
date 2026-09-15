from timm.layers import DropPath
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.logger import print_
from timm.models.vision_transformer import Mlp, PatchEmbed
from einops import repeat, rearrange


try:
    import xformers.ops
except ImportError:
    xformers = None

from models.model_blocks import TemporalPositionalEncoding
from lib.rope import RotaryEmbedding
from models.embeddings import StochasticTimeEmbedding, RotaryEmbedding3D

def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift

class CaptionEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU(approximate='tanh'), token_num=120):
        super().__init__()
        self.y_proj = Mlp(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0)
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1

        null_embeds = self.y_embedding.unsqueeze(0).expand(caption.shape[0], -1, -1)
        caption = torch.where(drop_ids[:, None, None], null_embeds, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[1:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        rope: nn.Module = None,
        fused_attn: bool = True,
        using_slots: bool = False
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.using_slots = using_slots

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            if not self.using_slots:
                q = self.rope(q)
                k = self.rope(k)
            else:
                q, k = self.rope.rotate_qk(query=q, key=k, start=0, time_dim=2)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TemporalRoPEAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_slots: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        rope: nn.Module = None,
        fused_attn: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn
        self.num_slots = num_slots

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    def forward(self, x: torch.Tensor, T: int = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None and T is not None:
            q = q.view(B, self.num_heads, T, self.num_slots, self.head_dim)
            k = k.view(B, self.num_heads, T, self.num_slots, self.head_dim)

            q = self.rope.rotate(q, start=0, time_dim=2)
            k = self.rope.rotate(k, start=0, time_dim=2, invert_decay=True)

            q = q.view(B, self.num_heads, N, self.head_dim)
            k = k.view(B, self.num_heads, N, self.head_dim)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0., proj_drop=0., **block_kwargs):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model*2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        if q.is_cuda and xformers is not None:
            attn_bias = None
            if mask is not None:
                with torch.cuda.device(q.device):
                    attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
            x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p if self.training else 0.0, attn_bias=attn_bias)
        else:
            lengths = mask if mask is not None else [k.shape[1]]
            queries = q.split(N, dim=1) if mask is not None else [q]
            outputs = [F.scaled_dot_product_attention(qi.transpose(1, 2), ki.transpose(1, 2), vi.transpose(1, 2),
                       dropout_p=self.attn_drop.p if self.training else 0.0).transpose(1, 2)
                       for qi, ki, vi in zip(queries, k.split(lengths, dim=1), v.split(lengths, dim=1))]
            x = torch.cat(outputs, dim=1)
        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class DFoTBlock_New(nn.Module):
    def __init__(self, hidden_size, num_heads, num_slots=8, mlp_ratio=4.0, layer_norm_ca=True, rope=None, preserve_slot_equivariance=True, **block_kwargs):
        super().__init__()
        self.drop_path = DropPath(block_kwargs.get("drop_path", 0.0))
        self.use_qk_norm = block_kwargs.get('use_qk_norm', False)
        self.preserve_slot_equivariance = preserve_slot_equivariance

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if preserve_slot_equivariance:
            print_("--> Preserving slot permutation equivariance using TemporalRoPEAttention in DiTBlock")
            self.attn = TemporalRoPEAttention(
                hidden_size,
                num_heads=num_heads,
                num_slots=num_slots,
                qkv_bias=True,
                qk_norm= self.use_qk_norm,
                rope=rope
            )
        else:
            print_("--> Using regular Attention in DiTBlock, which does not preserve slot permutation equivariance")
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=self.use_qk_norm, rope=rope, using_slots=True)

        self.layer_norm_ca = layer_norm_ca
        if layer_norm_ca:
            self.norm_ca_text_q = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm_ca_text_kv = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn_text = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.adaln_cross_attention = block_kwargs.get('adaln_cross_attention', False)
        if self.adaln_cross_attention:
            print_("--> Using adaLN modulation for cross-attention in DiTBlock")
            self.adaLN_modulation_ca = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 3 * hidden_size, bias=True)
            )

    def forward(self, x, y, t, mask=None):
        B, T, num_slots, hidden_size = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=2)
        if self.adaln_cross_attention:
            shift_ca_text, scale_ca_text, gate_ca_text = self.adaLN_modulation_ca(t).chunk(3, dim=2)

        x = x.reshape(B, T*num_slots, hidden_size)
        if self.preserve_slot_equivariance:
            x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), T=T))
        else:
            x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)))

        if self.layer_norm_ca:
            x_norm = self.norm_ca_text_q(x)
            y_norm = self.norm_ca_text_kv(y)
        if self.adaln_cross_attention:
            x = x + self.drop_path(gate_ca_text * self.cross_attn_text(t2i_modulate(x_norm, shift_ca_text, scale_ca_text), y_norm, mask))
        else:
            x = x + self.drop_path(self.cross_attn_text(x_norm, y_norm, mask))

        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        x = x.reshape(B, T, num_slots, hidden_size)
        return x

class DFoTBlock_VAE(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, layer_norm_ca=True, rope=None, **block_kwargs):
        super().__init__()
        self.drop_path = DropPath(block_kwargs.get("drop_path", 0.0))
        self.use_qk_norm = block_kwargs.get('use_qk_norm', False)

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=self.use_qk_norm, rope=rope)

        self.layer_norm_ca = layer_norm_ca
        if layer_norm_ca:
            self.norm_ca_text_q = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm_ca_text_kv = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn_text = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        self.adaln_cross_attention = block_kwargs.get('adaln_cross_attention', False)
        if self.adaln_cross_attention:
            print_("--> Using adaLN modulation for cross-attention in DiTBlock")
            self.adaLN_modulation_ca = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 3 * hidden_size, bias=True)
            )

    def forward(self, x, y, t, mask=None):
        B, T, num_patches, hidden_size = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=2)
        if self.adaln_cross_attention:
            shift_ca_text, scale_ca_text, gate_ca_text = self.adaLN_modulation_ca(t).chunk(3, dim=2)

        x = x.reshape(B, T*num_patches, hidden_size)
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)))

        if self.layer_norm_ca:
            x_norm = self.norm_ca_text_q(x)
            y_norm = self.norm_ca_text_kv(y)
        if self.adaln_cross_attention:
            x = x + self.drop_path(gate_ca_text * self.cross_attn_text(t2i_modulate(x_norm, shift_ca_text, scale_ca_text), y_norm, mask))
        else:
            x = x + self.drop_path(self.cross_attn_text(x_norm, y_norm, mask))

        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        x = x.reshape(B, T, num_patches, hidden_size)
        return x

class FinalLayerDFoT(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class FinalLayerDFoT_VAE(nn.Module):
    def __init__(self, hidden_size, out_channels, patch_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class DFoT(nn.Module):
    def __init__(
        self,
        num_slots=8,
        input_dim=128,
        hidden_size=1152,
        depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        text_dropout_prob=0.1,
        learn_sigma=True,
        buffer_size=10,
        caption_channels=512,
        model_max_length=120,
        layer_norm_ca=False,
        pooling=False,
        zero_init=True,
        use_rope=False,
        use_temp_attn_layers=True,
        **kwargs
    ):
        super().__init__()
        self.num_slots = num_slots
        self.learn_sigma = learn_sigma
        self.in_channels = input_dim
        self.out_channels = input_dim * 2 if learn_sigma else input_dim
        self.num_heads = num_heads
        self.pooling = pooling
        self.buffer_size = buffer_size
        self.max_tokens = self.buffer_size * self.num_slots
        self.use_temp_attn_layers = use_temp_attn_layers
        self.zero_init = zero_init
        if self.zero_init:
            print_("    --> Initializing DFoT with zero initialization.")
        else:
            print_("    --> Not using zero initialization for DFoT.")

        if self.pooling:
            print_("    --> Using pooling: adding pooled text embeddings to time embedding for global adaLN modulation.")

        self.use_rope = use_rope
        self.rope = None
        if self.use_rope:
            print_("    --> Using RoPE in DFoT.")
            self.rope = RotaryEmbedding(
                dim=hidden_size // num_heads,
            )
        else:
            print_("    --> Using absolute positional encoding for the input in DFoT.")
            #NOTE: not tried.
            self.pe_input = TemporalPositionalEncoding(
                    d_model=hidden_size,
                    max_len=self.buffer_size + 1,
                    mode="learned"
                )

        self.mlp_in = nn.Linear(input_dim, hidden_size, bias=True)
        self.t_embedder = StochasticTimeEmbedding(
            dim=max(hidden_size // 4, 32),
            time_embed_dim=hidden_size,
            use_fourier=False,
        )
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size, uncond_prob=text_dropout_prob, act_layer=approx_gelu, token_num=model_max_length)

        if self.use_temp_attn_layers:
            print_("    --> Using temporal attention layers in DFoT.")

        else:
            print_("    --> Not using temporal attention layers in DFoT.")
            self.blocks = nn.ModuleList([
                DFoTBlock_New(hidden_size, num_heads, num_slots, mlp_ratio=mlp_ratio, layer_norm_ca=layer_norm_ca, rope=self.rope, **kwargs) for _ in range(depth)
            ])

        self.final_layer = FinalLayerDFoT(hidden_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        def _mlp_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.t_embedder.apply(_mlp_init)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        if self.zero_init:
            for block in self.blocks:
                nn.init.constant_(block.cross_attn_text.proj.weight, 0)
                nn.init.constant_(block.cross_attn_text.proj.bias, 0)
                if hasattr(block, "temp_attn"):
                    nn.init.constant_(block.temp_attn.proj.weight, 0)
                    nn.init.constant_(block.temp_attn.proj.bias, 0)

        if hasattr(self.final_layer, "adaLN_modulation"):
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        if self.zero_init:
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, y, mask=None):
        x = self.mlp_in(x)
        if not self.use_rope:
            x = self.pe_input(x=x, batch_size=x.shape[0], num_slots=x.shape[-2])

        y = self.y_embedder(y, self.training)

        t = self.t_embedder(t)

        if self.pooling:
            y_pooled = y.mean(dim=1, keepdim=True)
            t = t + y_pooled

        t = repeat(t, "b t c -> b (t p) c", p=self.num_slots)

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            y = y.masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.bool().sum(dim=1).tolist()
        else:
            y_lens = [y.shape[1]] * y.shape[0]
            y = y.view(1, -1, x.shape[-1])

        for block in self.blocks:
            x = block(x, y, t, y_lens)

        B, T, num_slots, D = x.shape
        x = x.reshape(B, T*num_slots, D)
        x = self.final_layer(x, t)
        x = x.reshape(B, T, num_slots, self.out_channels)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, mask=None):
        if t.dim() == 1:
            t = t.unsqueeze(1).expand(-1, x.shape[1])
        model_out = self.forward(x, t, y, mask)

        eps, rest = model_out[:, :, :, :self.in_channels], model_out[:, :, :, self.in_channels:]

        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=3)

    def forward_with_fractional_hg(self, x, t, y, cfg_scale, mask=None):
        original_batch_size = x.shape[0] // 3

        if t.dim() == 1:
            t = t.unsqueeze(1).expand(-1, x.shape[1])

        model_out_full_conditioning = self.forward(x[:original_batch_size], t[:original_batch_size], y[:original_batch_size], mask)

        model_out_no_conditioning = self.forward(x[original_batch_size:2*original_batch_size], t[original_batch_size:2*original_batch_size], y[original_batch_size:2*original_batch_size], mask)

        model_out_partial_conditioning = self.forward(x[2*original_batch_size:], t[2*original_batch_size:], y[2*original_batch_size:], mask)

        cond_eps = model_out_full_conditioning[:, :, :, :self.in_channels]
        cond_rest = model_out_full_conditioning[:, :, :, self.in_channels:]
        uncond_eps = model_out_no_conditioning[:, :, :, :self.in_channels]
        uncond_rest = model_out_no_conditioning[:, :, :, self.in_channels:]
        partial_cond_eps = model_out_partial_conditioning[:, :, :, :self.in_channels]
        partial_cond_rest = model_out_partial_conditioning[:, :, :, self.in_channels:]

        half_eps = cond_eps + cfg_scale * (partial_cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps, half_eps], dim=0)
        rest = torch.cat([cond_rest, uncond_rest, partial_cond_rest], dim=0)
        return torch.cat([eps, rest], dim=3)

class DFoT_VAE(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        text_dropout_prob=0.1,
        learn_sigma=True,
        buffer_size=10,
        caption_channels=512,
        model_max_length=120,
        layer_norm_ca=False,
        pooling=False,
        zero_init=True,
        use_rope=False,
        use_temp_attn_layers=False,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_patches = (input_size // self.patch_size) ** 2
        self.spatial_size = input_size // patch_size

        self.pooling = pooling
        self.buffer_size = buffer_size
        self.max_tokens = self.buffer_size * self.num_patches
        self.use_temp_attn_layers = use_temp_attn_layers
        self.zero_init = zero_init
        if self.zero_init:
            print_("    --> Initializing DFoT with zero initialization.")
        else:
            print_("    --> Not using zero initialization for DFoT.")

        if self.pooling:
            print_("    --> Using pooling: adding pooled text embeddings to time embedding for global adaLN modulation.")

        self.use_rope = use_rope
        self.rope = None
        if self.use_rope:
            print_("    --> Using 3D RoPE in DFoT_VAE.")
            self.rope = RotaryEmbedding3D(
                dim=hidden_size // num_heads,
                sizes=(
                    self.buffer_size,
                    self.spatial_size,
                    self.spatial_size,
                ),
            )
        else:
            print_("    --> Using absolute positional encoding for the input in DFoT.")
            #NOTE: not tried.
            self.pe_input = TemporalPositionalEncoding(
                    d_model=hidden_size,
                    max_len=self.max_tokens,
                    mode="learned"
                )

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)

        self.t_embedder = StochasticTimeEmbedding(
            dim=max(hidden_size // 4, 32),
            time_embed_dim=hidden_size,
            use_fourier=False,
        )
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size, uncond_prob=text_dropout_prob, act_layer=approx_gelu, token_num=model_max_length)

        if self.use_temp_attn_layers:
            print_("    --> Using temporal attention layers in DFoT.")

        else:
            print_("    --> Not using temporal attention layers in DFoT.")
            self.blocks = nn.ModuleList([
                DFoTBlock_VAE(hidden_size, num_heads, mlp_ratio=mlp_ratio, layer_norm_ca=layer_norm_ca, rope=self.rope, **kwargs) for _ in range(depth)
            ])

        self.final_layer = FinalLayerDFoT_VAE(hidden_size, self.out_channels, patch_size=patch_size)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        def _mlp_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.t_embedder.apply(_mlp_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        if self.zero_init:
            for block in self.blocks:
                nn.init.constant_(block.cross_attn_text.proj.weight, 0)
                nn.init.constant_(block.cross_attn_text.proj.bias, 0)
                if hasattr(block, "temp_attn"):
                    nn.init.constant_(block.temp_attn.proj.weight, 0)
                    nn.init.constant_(block.temp_attn.proj.bias, 0)

        if hasattr(self.final_layer, "adaLN_modulation"):
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        if self.zero_init:
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(
            x,
            "b (h w) (p q c) -> b (h p) (w q) c",
            h=int(self.num_patches**0.5),
            p=self.patch_size,
            q=self.patch_size,
        )

    def forward(self, x, t, y, mask=None):
        input_batch_size = x.shape[0]
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = self.x_embedder(x)

        x = rearrange(x, "(b t) p c -> b t p c", b=input_batch_size)
        if not self.use_rope:
            x = self.pe_input(x=x, batch_size=x.shape[0], num_slots=x.shape[-2])

        t = self.t_embedder(t)

        y = self.y_embedder(y, self.training)

        if self.pooling:
            y_pooled = y.mean(dim=1, keepdim=True)
            t = t + y_pooled

        t = repeat(t, "b t c -> b (t p) c", p=self.num_patches)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            y = y.masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.bool().sum(dim=1).tolist()
        else:
            y_lens = [y.shape[1]] * y.shape[0]
            y = y.view(1, -1, x.shape[-1])

        for block in self.blocks:
            x = block(x, y, t, y_lens)

        B, T, num_patches, D = x.shape
        x = x.reshape(B, T*num_patches, D)
        x = self.final_layer(x, t)

        x = self.unpatchify(
            rearrange(x, "b (t p) c -> (b t) p c", p=self.num_patches)
        )

        x = rearrange(
            x, "(b t) h w c -> b t c h w", b=input_batch_size
        )

        return x
