import logging
import argparse
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F

import data_utils.generate_parabola as gp

import generators.vae as vae


# Constants
VECTOR_SIZE = 10
BATCH_SIZE = 2
NUM_HEADS = 2
NUM_LAYERS = 2
DIM_FEEDFORWARD = 50
LATENT_DIM = 2

NUM_EPOCHS = 10
SAVE_EPOCHS = 1


vae_ckpts_dir = Path("/workspaces/GenAI-101/models/vae_ckpts")

load_ckpt_path = vae_ckpts_dir / "checkpoint_epoch_5000.pth"

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input", type=str, required=True)
    
    run_training()
    # run_inference()


def run_inference():

    # Example of loading a checkpoint
    model = vae.VAE(VECTOR_SIZE, LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    start_epoch, previous_loss = vae.load_checkpoint(str(load_ckpt_path), model, optimizer)

    print('prev epoc', start_epoch)
    print('prev loss', previous_loss)

    num_samples = 5    
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, LATENT_DIM)
        samples = model.decode(z)

    print(samples)


def run_training():
    # PREPARE data set
    from torch.utils.data import DataLoader, TensorDataset
    data = torch.tensor(gen_parabola_data(sample_count=100), dtype=torch.float32)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)


    # import generators.tranformer as tf
    # model = tf.MiniTransformer(VECTOR_SIZE, NUM_HEADS, NUM_LAYERS, DIM_FEEDFORWARD)

    model = vae.VAE(VECTOR_SIZE, LATENT_DIM)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # TRAINING LOOP
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0

        for batch, (inputs,) in enumerate(dataloader):
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(inputs)
            loss = vae.loss_function(recon_batch, inputs, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {train_loss / len(dataloader)}")

        if (epoch + 1) % SAVE_EPOCHS == 0:  # Checkpoint every SAVE_EPOCHS epochs
            vae_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }

            out_ckpt_path = vae_ckpts_dir / f"checkpoint_epoch_{epoch+1}.pth"
            
            vae.save_checkpoint(vae_model_state, filename=str(out_ckpt_path))
            


def gen_parabola_data(sample_count=100):
    line_count = sample_count
    a, b, c = gp.generate_parabola(0, 0, line_count)  # a.shape = (line_count,), same for b, c

    x_count = 10
    x_values = np.linspace(0, x_count - 1, x_count)  # x_values.shape = (x_count,)

    y_values = gp.get_parabola_y_values(a, b, c, x_values)  # y_values.shape = (line_count, x_count)

    # plot_parabola(x_values, y_values)
    return y_values


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
