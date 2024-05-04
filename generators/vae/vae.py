import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Constants
VECTOR_SIZE = 10
BATCH_SIZE = 5
LATENT_DIM = 2
EPOCHS = 50

# VAE Model
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc21 = nn.Linear(50, latent_dim)  # Mean of the latent space
        self.fc22 = nn.Linear(50, latent_dim)  # Log variance of the latent space
        self.fc3 = nn.Linear(latent_dim, 50)
        self.fc4 = nn.Linear(50, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    # def decode(self, z):
    #     h3 = F.relu(self.fc3(z))
    #     return torch.sigmoid(self.fc4(h3))
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)  # No sigmoid if using MSE and data isn't in [0,1]
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, VECTOR_SIZE))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Loss Function
def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, VECTOR_SIZE), reduction='sum')
    MSE = F.mse_loss(recon_x, x.view(-1, VECTOR_SIZE), reduction='sum')

    # KLD is Kullback-Leibler divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


# vae_model_state = {
#     'epoch': epoch + 1,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': train_loss,
# }

def save_checkpoint(state, filename="checkpoint.pth"):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(filepath, model, optimizer):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss}")
    return epoch, loss
