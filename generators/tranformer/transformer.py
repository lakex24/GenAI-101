import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = \
            nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask):
        x2 = self.norm1(x)
        x, _ = self.attention(x2, x2, x2, attn_mask=mask)
        x = x + x2
        x2 = self.norm2(x)
        x = x + self.feed_forward(x2)
        return x

class MiniTransformer(nn.Module):
    def __init__(self, dim, heads, depth, seq_length, num_tokens):
        super(MiniTransformer, self).__init__()
        self.layers = nn.ModuleList([])
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, dim))
        for _ in range(depth):
            self.layers.append(TransformerBlock(dim, heads))
        self.decoder = nn.Linear(dim, num_tokens)

    def forward(self, x, mask=None):
        x += self.pos_embedding
        for layer in self.layers:
            x = layer(x, mask)
        x = self.decoder(x)
        return x
