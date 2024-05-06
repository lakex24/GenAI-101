import math

import torch
import torch.nn as nn


class MiniTransformer(nn.Module):
    """
    # Mini Transformer Model
    """
    def __init__(self, vector_size, num_heads, num_layers, dim_feedforward):

        # dim_feedforward, is the dimension of the feedforward network model

        super(MiniTransformer, self).__init__()
        self.embed = nn.Linear(vector_size, vector_size)
        self.pos_encoder = PositionalEncoding(vector_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=vector_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward
        )
        
        self.transformer_encoder = \
            nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Linear(vector_size, vector_size)

    def forward(self, x_in):
        x = self.embed(x_in)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        output = self.decoder(output)
        return output


# trans_model_state = {
#     'epoch': epoch + 1,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': train_loss,
# }

def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


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


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):
        super(PositionalEncoding, self).__init__()

        # d_model (vector_size) is the number of expected features in the input
        # max_len is the maximum length of the input sequence
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)]

# NEW CODE::::
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', self.encoding)

    def forward(self, x):
        # x is expected to have dimension [batch size, sequence length, model dimension]
        return x + self.pe[:x.size(1), :]

# Example use
d_model = 10  # Dimension of the model (must match the dimension of the data vector)
max_len = 10  # Maximum length of the input sequences
pos_encoder = PositionalEncoding(d_model, max_len)

# Create a batch of input data vectors
batch_size = 2
input_data = torch.randn(batch_size, max_len, d_model)

# Apply positional encoding
encoded_data = pos_encoder(input_data)
print("Input Data Shape:", input_data.shape)
print("Encoded Data Shape:", encoded_data.shape)
