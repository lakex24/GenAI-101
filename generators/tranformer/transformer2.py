import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        """
        model_dim: the number of expected features in the model
        """

        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.input_linear = nn.Linear(input_dim, model_dim)  # Linear transformation for input
        self.pos_encoder = nn.Parameter(torch.randn(1, 10, model_dim))  # Positional encoding
        encoder_layers = nn.TransformerEncoderLayer(model_dim, num_heads, model_dim * 2, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(model_dim, input_dim)  # Linear transformation to original space

    def forward(self, x):
        src = self.input_linear(x) + self.pos_encoder
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output


def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")
