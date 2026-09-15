import torch
import torch.nn as nn

class TransformerTextEncoder(nn.Module):
    def __init__(self, input_dim, num_layers, num_heads, output_dim, vocab_size=50,
                 context_length=50, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = 0

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.token_embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=input_dim,
                padding_idx=self.padding_idx
            )
        self.position_embedding = nn.Embedding(
                num_embeddings=context_length,
                embedding_dim=input_dim
            )
        self.layer_norm = nn.LayerNorm(input_dim, eps=1e-8, elementwise_affine=True)
        self.dropout = nn.Dropout(p=dropout)
        self.text_out_projection = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim)
            )

        self.apply(self._init_weights)
        return

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        return

    def forward(self, text, text_length):
        position_indices = self._create_position_indices(text)
        text_tokens = self.token_embedding(text)
        position_embeddings = self.position_embedding(position_indices)
        text_tokens = self.layer_norm(text_tokens + position_embeddings)
        text_tokens = self.dropout(text_tokens)

        token_mask = (text != self.padding_idx).unsqueeze(-1)
        text_tokens = text_tokens * token_mask.type(text_tokens.dtype)
        ones = torch.ones_like(text)
        caption_mask = text_length.unsqueeze(1) < ones.cumsum(dim=1)

        text_tokens = text_tokens.permute(1, 0, 2)
        text_embeddings = self.transformer(
                text_tokens,
                mask=None,
                src_key_padding_mask=caption_mask,
            )
        text_embeddings = text_embeddings.permute(1, 0, 2)

        text_embeddings = self.text_out_projection(text_embeddings)
        return text_embeddings

    def _create_position_indices(self, tokens):
        batch_size, max_caption_length = tokens.size()
        positions = torch.arange(max_caption_length, dtype=tokens.dtype, device=tokens.device)
        positions = positions.view(1, max_caption_length).repeat(batch_size, 1)
        return positions
