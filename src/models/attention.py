import torch
import torch.nn as nn

from models.model_utils import init_xavier_
from lib.utils import get_from_dict
from models.text_modules import TransformerTextEncoder

class SlotAttention(nn.Module):
    def __init__(self, dim_feats, dim_slots, num_slots, num_iters_first=2, num_iters=2,
                 mlp_hidden=128, epsilon=1e-8):
        super().__init__()
        self.dim_slots = dim_slots
        self.num_iters_first = num_iters_first
        self.num_iters = num_iters
        self.num_slots = num_slots
        self.epsilon = epsilon
        self.scale = dim_feats ** -0.5

        self.norm_input = nn.LayerNorm(dim_feats, eps=0.001)
        self.norm_slot = nn.LayerNorm(dim_slots, eps=0.001)
        self.norm_mlp = nn.LayerNorm(dim_slots, eps=0.001)

        self.to_q = nn.Linear(dim_slots, dim_slots)
        self.to_k = nn.Linear(dim_feats, dim_slots)
        self.to_v = nn.Linear(dim_feats, dim_slots)

        self.gru = nn.GRUCell(dim_slots, dim_slots)
        self.mlp = nn.Sequential(
            nn.Linear(dim_slots, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, dim_slots),
        )
        return

    def forward(self, inputs, slots, step=0, **kwargs):
        B, N, D = inputs.shape
        self.attention_masks = None

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        num_iters = self.num_iters_first if step == 0 else self.num_iters
        for _ in range(num_iters):
            slots_prev = slots
            slots = self.norm_slot(slots)
            q = self.to_q(slots)

            dots = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.epsilon
            attn = attn / attn.sum(dim=-1, keepdim=True)
            self.attention_masks = attn
            updates = torch.einsum('b i d , b d j -> b i j', attn, v)

            slots = self.gru(
                updates.reshape(-1, self.dim_slots),
                slots_prev.reshape(-1, self.dim_slots)
            )
            slots = slots.reshape(B, -1, self.dim_slots)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots

    def get_attention_masks(self):
        B, N_slots, N_locs = self.attention_masks.shape
        masks = self.attention_masks
        return masks

class MetaAttention(nn.Module):
    def __init__(self, emb_dim, num_heads=1, dropout=0., out_dim=None, **kwargs):
        assert num_heads >= 1
        assert emb_dim % num_heads == 0, "Embedding dim. must be divisible by number of heads..."
        super().__init__()

        out_dim = out_dim if out_dim is not None else emb_dim
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        self.q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v = nn.Linear(emb_dim, emb_dim, bias=False)
        self.drop = nn.Dropout(dropout)

        self.out_projection = nn.Sequential(
                nn.Linear(emb_dim, out_dim, bias=False)
            )
        self.attention_masks = None
        return

    def forward(self, x):
        raise NotImplementedError("Base-Class does not implement a 'forward' method...")

    def attention(self, query, key, value, dim_head, mask=None, alibi_mask=None, **kwargs):
        scale = dim_head ** -0.5
        dots = torch.einsum('b i d , b j d -> b i j', query, key) * scale

        if mask is not None:
            dots = dots.masked_fill(mask, float('-inf'))

        if alibi_mask is not None:
            dots = dots + alibi_mask

        attention = dots.softmax(dim=-1)
        self.attention_masks = attention
        attention = self.drop(attention)
        vect = torch.einsum('b i d , b d j -> b i j', attention, value)

        return vect

    def get_attention_masks(self, reshape=None):
        assert self.attention_masks is not None, "Attention masks have not yet been computed..."
        masks = self.attention_masks
        return masks

    def split_into_heads(self, x):
        batch_size, num_tokens, token_dim = x.shape

        dim_head = token_dim // self.num_heads

        x = x.view(batch_size, num_tokens, self.num_heads, dim_head).transpose(1, 2)
        y = x.reshape(batch_size * self.num_heads, num_tokens, dim_head)
        return y

    def merge_heads(self, x):
        _, num_tokens, dim_head = x.shape
        x = x.reshape(-1, self.num_heads, num_tokens, dim_head).transpose(1, 2)
        y = x.reshape(-1, num_tokens, self.num_heads * dim_head)
        return y

class MultiHeadSelfAttention(MetaAttention):
    def __init__(self, emb_dim, num_heads=8, dropout=0.):
        super().__init__(
                emb_dim=emb_dim,
                num_heads=num_heads,
                dropout=dropout
            )
        return

    def forward(self, x, **kwargs):
        batch_size, num_tokens, token_dim = x.size()
        dim_head = token_dim // self.num_heads
        mask = kwargs.get("mask", None)
        alibi_mask = kwargs.get("alibi_mask", None)

        q, k, v = self.q(x), self.k(x), self.v(x)

        q = q.view(batch_size, num_tokens, self.num_heads, dim_head).transpose(1, 2)
        q = q.reshape(batch_size * self.num_heads, num_tokens, dim_head)
        k = k.view(batch_size, num_tokens, self.num_heads, dim_head).transpose(1, 2)
        k = k.reshape(batch_size * self.num_heads, num_tokens, dim_head)
        v = v.view(batch_size, num_tokens, self.num_heads, dim_head).transpose(1, 2)
        v = v.reshape(batch_size * self.num_heads, num_tokens, dim_head)

        vect = self.attention(query=q, key=k, value=v, dim_head=dim_head, mask=mask, alibi_mask=alibi_mask)

        vect = vect.reshape(batch_size, self.num_heads, num_tokens, dim_head).transpose(1, 2)
        vect = vect.reshape(batch_size * num_tokens, self.num_heads * dim_head)

        y = self.out_projection(vect)
        y = y.reshape(batch_size, num_tokens, self.num_heads * dim_head)
        return y

class MultiHeadCrossAttention(MetaAttention):
    def __init__(self, emb_dim, dim_head, kv_dim, num_heads=8, dropout=0.):
        super().__init__(
                emb_dim=emb_dim,
                num_heads=num_heads,
                dropout=dropout
            )

        self.dim_head = dim_head

        inner_dim = dim_head * num_heads
        self.q = nn.Linear(emb_dim, inner_dim, bias=False)
        self.k = nn.Linear(kv_dim, inner_dim, bias=False)
        self.v = nn.Linear(kv_dim, inner_dim, bias=False)

        self.out_projection = nn.Linear(inner_dim, emb_dim)

        return

    def forward(self, enc_embs, query_embs, **kwargs):
        batch_size, num_tokens, token_dim = enc_embs.shape
        _, num_queries, query_dim = query_embs.shape

        q, k, v = self.q(query_embs), self.k(enc_embs), self.v(enc_embs)
        q = self.split_into_heads(q)
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        vect = self.attention(query=q, key=k, value=v, dim_head=self.dim_head)

        y = self.merge_heads(vect)
        y = self.out_projection(y)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_size, pre_norm=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.pre_norm = pre_norm
        assert num_heads >= 1

        self.attn = MultiHeadSelfAttention(
            emb_dim=embed_dim,
            num_heads=num_heads,
        )

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, embed_dim),
        )

        self.layernorm_query = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_mlp = nn.LayerNorm(embed_dim, eps=1e-6)
        self._init_model()
        return

    @torch.no_grad()
    def _init_model(self):
        init_xavier_(self)

    def forward(self, inputs):
        assert inputs.ndim == 3
        B, L, _ = inputs.shape

        if self.pre_norm:
            x = self.layernorm_query(inputs)
            x = self.attn(x)
            x = x + inputs

            y = x

            z = self.layernorm_mlp(y)
            z = self.mlp(z)
            z = z + y
        else:
            x = self.attn(inputs)
            x = x + inputs
            x = self.layernorm_query(x)

            y = x

            z = self.mlp(y)
            z = z + y
            z = self.layernorm_mlp(z)

        return z

class AdaptedEncoderBlock_new(TransformerBlock):
    def __init__(self, embed_dim, num_heads, mlp_size, fusion_params, text_encoder_params, pre_norm=False):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_size=mlp_size,
            pre_norm=pre_norm
        )

        self.ln_cross_att_q = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln_cross_att_kv = nn.LayerNorm(embed_dim, eps=1e-6)
        self.cross_attention = MultiHeadCrossAttention(
            emb_dim=embed_dim,
            kv_dim=embed_dim,
            dim_head=get_from_dict(fusion_params, "head_dim"),
            num_heads=get_from_dict(fusion_params, "num_heads"),
        )

        return

    def forward(self, all_inputs):
        inputs = get_from_dict(all_inputs, "inputs")
        alibi_mask = get_from_dict(all_inputs, "alibi_mask", default=None)

        assert inputs.ndim == 3
        B, L, _ = inputs.shape

        text_embeddings = get_from_dict(all_inputs, "text_embeddings").to(inputs.device)

        x = self.layernorm_query(inputs)
        x = self.attn(x, alibi_mask=alibi_mask)
        x = x + inputs

        query_embs = self.ln_cross_att_q(x)
        kv_embs = self.ln_cross_att_kv(text_embeddings)
        z = self.cross_attention(kv_embs, query_embs=query_embs)
        z = z + x

        out = self.layernorm_mlp(z)
        out = self.mlp(out)
        out = out + z

        all_inputs['inputs'] = out

        return all_inputs

class AdaptedEncoderBlock(TransformerBlock):
    def __init__(self, embed_dim, num_heads, mlp_size, fusion_params, text_encoder_params, pre_norm=False):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_size=mlp_size,
            pre_norm=pre_norm
        )

        self.cross_attention = TransformerDecoderBlock(
            embed_dim=embed_dim,
            kv_dim=embed_dim,
            head_dim=get_from_dict(fusion_params, "head_dim"),
            num_heads=get_from_dict(fusion_params, "num_heads"),
            mlp_size=get_from_dict(fusion_params, "mlp_size")
        )

        return

    def forward(self, all_inputs):
        inputs = get_from_dict(all_inputs, "inputs")
        alibi_mask = get_from_dict(all_inputs, "alibi_mask", default=None)

        assert inputs.ndim == 3
        B, L, _ = inputs.shape

        text_embeddings = get_from_dict(all_inputs, "text_embeddings").to(inputs.device)

        if self.pre_norm:
            x = self.layernorm_query(inputs)
            x = self.attn(x, alibi_mask=alibi_mask)
            x = x + inputs

            y = x

            z = self.condition_slots_given_caption(
                slots_to_condition=y,
                text_embeddings=text_embeddings
            )

            z = self.layernorm_mlp(z)
            z = self.mlp(z)
            z = z + y
        else:
            x = self.attn(x, alibi_mask=alibi_mask)
            x = x + inputs
            x = self.layernorm_query(x)

            y = x

            z = self.condition_slots_given_caption(
                slots_to_condition=y,
                text_embeddings=text_embeddings
            )

            z = self.mlp(z)
            z = z + y
            z = self.layernorm_mlp(z)

        all_inputs['inputs'] = z

        return all_inputs

    def condition_slots_given_caption(self, slots_to_condition, text_embeddings):
        conditioned_slots = self.cross_attention(queries=slots_to_condition, feats=text_embeddings)

        return conditioned_slots

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, head_dim, kv_dim, num_heads, mlp_size):
        super().__init__()

        self.ln_mlp = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, embed_dim),
        )

        self.ln_cross_att_q = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ln_cross_att_kv = nn.LayerNorm(kv_dim, eps=1e-6)
        self.cross_attn = MultiHeadCrossAttention(
                emb_dim=embed_dim,
                dim_head=head_dim,
                num_heads=num_heads,
                kv_dim=kv_dim
            )
        return

    def forward(self, queries, feats):
        assert queries.ndim == 3
        B, L, _ = queries.shape

        query_embs = self.ln_cross_att_q(queries)
        feats = self.ln_cross_att_kv(feats)
        z = self.cross_attn(feats, query_embs=query_embs)
        z = z + queries

        out = self.ln_mlp(z)
        out = self.mlp(out)
        out = out + z

        return out

    def get_attention_masks(self, reshape=None):
        return self.cross_attn.get_attention_masks(reshape=reshape)

class OriginalTransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, head_dim, num_heads, mlp_size, kv_dim=None, dropout=0,
                 use_cross_attn=False, project_out=False):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        self.ln_att = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(
                emb_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )

        if use_cross_attn:
            assert kv_dim is not None, f"If {use_cross_attn = }, 'kv_dim' must be provided..."
            self.ln_cross_att_q = nn.LayerNorm(embed_dim, eps=1e-6)
            self.ln_cross_att_kv = nn.LayerNorm(kv_dim, eps=1e-6)
            self.cross_attn = MultiHeadCrossAttention(
                    emb_dim=embed_dim,
                    dim_head=head_dim,
                    num_heads=num_heads,
                    kv_dim=kv_dim,
                    dropout=dropout
                )

        self.ln_mlp = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, embed_dim),
        )

        return

    def forward(self, queries, feats=None, self_attn_mask=None, cross_attn_mask=None):
        assert queries.ndim == 3
        B, L, _ = queries.shape

        x = self.ln_att(queries)
        x = self.attn(x, mask=self_attn_mask)
        y = x + queries

        if self.use_cross_attn:
            assert feats is not None, f"If {self.use_cross_attn = }, 'feats' must be provided"
            query_embs = self.ln_cross_att_q(y)
            feats = self.ln_cross_att_kv(feats)
            z = self.cross_attn(feats, query_embs=query_embs, mask=cross_attn_mask)
            z = z + y
        else:
            z = y

        out = self.ln_mlp(z)
        out = self.mlp(out)
        out = out + z

        return out

    def get_attention_masks(self, reshape=None):
        return self.cross_attn.get_attention_masks(reshape=reshape)

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=None, use_gelu=True):
        super().__init__()
        out_dim = out_dim if out_dim is not None else in_dim
        activation = nn.ReLU() if not use_gelu else nn.GELU()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        y = self.mlp(x)
        return y

class IDMAttention(nn.Module):
    def __init__(self, emb_dim, head_dim=None, num_heads=1, dropout=0., self_attn=True,
                 kv_dim=None, project_out=False, **kwargs):
        assert num_heads >= 1
        super().__init__()

        head_dim = head_dim if head_dim is not None else emb_dim
        inner_dim = num_heads * head_dim
        project_out = not inner_dim == emb_dim or project_out
        self.emb_dim = emb_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.inner_dim = inner_dim

        if self_attn:
            kv_dim = emb_dim
        elif kv_dim is None:
            raise ValueError(f"{kv_dim = } cannot be None in cross-attention mode")
        else:
            pass
        self.q = nn.Linear(emb_dim, inner_dim, bias=False)
        self.k = nn.Linear(kv_dim, inner_dim, bias=False)
        self.v = nn.Linear(kv_dim, inner_dim, bias=False)

        self.out_proj = nn.Sequential(
                nn.Linear(inner_dim, emb_dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()

        self.attention_masks = None
        return

    def forward(self, x):
        raise NotImplementedError("Base-Class does not implement a 'forward' method...")

    def attention(self, query, key, value, mask=None, **kwargs):
        scale = self.head_dim ** -0.5
        dots = torch.einsum('b i d , b j d -> b i j', query, key) * scale

        if mask is not None:
            dots = dots.masked_fill(mask == 0, -1e9)
        attention = dots.softmax(dim=-1)
        self.attention_masks = attention
        out = torch.einsum('b i j , b j d -> b i d', attention, value)
        return out

    def get_attention_masks(self, reshape=None):
        if self.attention_masks is None:
            raise ValueError("Attention masks have not yet been computed...")
        masks = self.attention_masks
        return masks

    def split_into_heads(self, x):
        batch_size, num_tokens, _ = x.shape
        x = x.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        y = x.reshape(batch_size * self.num_heads, num_tokens, self.head_dim)
        return y

    def merge_heads(self, x):
        _, num_tokens, dim_head = x.shape
        x = x.reshape(-1, self.num_heads, num_tokens, dim_head).transpose(1, 2)
        y = x.reshape(-1, num_tokens, self.num_heads * dim_head)
        return y

class IDMSelfAttention(IDMAttention):
    def __init__(self, emb_dim, head_dim, num_heads=8, dropout=0., project_out=False):
        super().__init__(
                emb_dim=emb_dim,
                head_dim=head_dim,
                num_heads=num_heads,
                self_attn=True,
                dropout=dropout,
                project_out=project_out
            )
        return

    def forward(self, x, **kwargs):
        q, k, v = self.q(x), self.k(x), self.v(x)
        q = self.split_into_heads(q)
        k = self.split_into_heads(k)
        v = self.split_into_heads(v)

        vect = self.attention(query=q, key=k, value=v, **kwargs)

        y = self.merge_heads(vect)
        y = self.out_proj(y)
        return y

class IDMTransformerBlock(nn.Module):
    def __init__(self, embed_dim, head_dim, num_heads, mlp_size, self_attn=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_size = mlp_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.self_attn = self_attn
        assert num_heads >= 1

        self.ln_mlp = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp = MLP(
            in_dim=embed_dim,
            hidden_dim=mlp_size,
        )
        return

    def forward(self, inputs):
        raise NotImplementedError("Base Class does not implement 'forward' function...")

    def get_attention_masks(self, reshape=None):
        attn_masks = self.attn.get_attention_masks(reshape=reshape)
        return attn_masks

class TransformerEncoderBlock(IDMTransformerBlock):
    def __init__(self, embed_dim, head_dim=32, num_heads=4, mlp_size=256,
                 self_attn=True, project_out=False):
        super().__init__(
                embed_dim=embed_dim,
                head_dim=head_dim,
                num_heads=num_heads,
                mlp_size=mlp_size,
                self_attn=True
            )

        self.ln_att = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = IDMSelfAttention(
                emb_dim=embed_dim,
                head_dim=head_dim,
                num_heads=num_heads,
                project_out=project_out
            )
        return

    def forward(self, inputs, mask=None):
        assert inputs.ndim == 3

        x = self.ln_att(inputs)
        x = self.attn(x, mask=mask)
        y = x + inputs

        z = self.ln_mlp(y)
        z = self.mlp(z)
        z = z + y
        return z
