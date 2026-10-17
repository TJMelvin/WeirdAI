import torch
from torch import nn

from weird_ai.layer_norm import LayerNorm
from weird_ai.feed_forward import FeedForward
from weird_ai.attention import CausalAttention


class TransformerBlock(nn.Module):

    def __init__(self, emb_dim, context_length, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()

        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)

        head_dim = emb_dim // num_heads

        self.attention_heads = nn.ModuleList([
            CausalAttention(
                embedding_dim=emb_dim,
                output_dim=head_dim,
                context_length=context_length,
                dropout=dropout,
                qkv_bias=qkv_bias
            )
            for _ in range(num_heads)
        ])

        self.feed_forward = FeedForward(emb_dim)

        self.dropout = nn.Dropout(dropout)

        self.output_projection = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)

        attention_outputs = [
            head(x) for head in self.attention_heads
        ]

        x = torch.cat(attention_outputs, dim=-1)
        x = self.output_projection(x)
        x = self.dropout(x)

        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)

        x = x + shortcut

        return x