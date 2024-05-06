import logging
import argparse
from pathlib import Path

import numpy as np

import torch
# import torch.nn as nn

import generators.tranformer as tf

import data_utils.generate_parabola as gp


# Constants
VECTOR_SIZE = 10
BATCH_SIZE = 8
NUM_HEADS = 2
NUM_LAYERS = 2
DIM_FEEDFORWARD = 50

NUM_EPOCHS = 1000
SAVE_EPOCHS = 200


trans_ckpts_dir = Path("/workspaces/GenAI-101/models/trans_ckpts")

# load_ckpt_path = trans_ckpts_dir / "checkpoint_epoch_5000.pth"


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input", type=str, required=True)
    
    # TODO: make a tiny transformer example work first !!!
    # TODO: maybe use sin/cos to generate wave data to feed into transformer?
    run_training()


def run_training():

    # PREPARE data set
    from torch.utils.data import DataLoader, TensorDataset
    data = torch.tensor(gen_parabola_data(sample_count=1000), dtype=torch.float32)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # noise_input_data = torch.randn(100, 10, dtype=torch.float32)

    # model = tf.MiniTransformer(VECTOR_SIZE, NUM_HEADS, NUM_LAYERS, DIM_FEEDFORWARD)
    model = tf.TransformerModel(input_dim=1, model_dim=8, num_heads=4, num_layers=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    # TRAINING LOOP
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0

        for idx, inputs in enumerate(dataloader):
            optimizer.zero_grad()
            
            # FIXME: x_in, feature values NOT very normalized?? ... might be the issue??
            x_in = inputs[0][:, :10].unsqueeze(-1)  # shape (1, 10, 1)
            outputs = model(x_in)

            x_out = inputs[0][:, 12:].unsqueeze(-1)  # shape (1, 10, 1)

            loss = torch.nn.MSELoss()(outputs[:,2:,:], x_out)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {train_loss / len(dataloader)}")
                
        if (epoch + 1) % SAVE_EPOCHS == 0:  # Checkpoint every SAVE_EPOCHS epochs
            trans_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss / len(dataloader),
            }

            out_ckpt_path = trans_ckpts_dir / f"checkpoint_epoch_{epoch+1}.pth"
            
            tf.save_checkpoint(trans_model_state, filename=str(out_ckpt_path))
            
            
def gen_parabola_data(sample_count=100):
    line_count = sample_count
    a, b, c = gp.generate_parabola(0, 0, line_count)  # a.shape = (line_count,), same for b, c

    x_count = 20
    # x_values = np.linspace(0, x_count - 1, x_count)  # x_values.shape = (x_count,)
    x_values = np.linspace(-x_count / 2 + 1, x_count / 2, x_count)  # x_values.shape = (x_count,)

    y_values = gp.get_parabola_y_values(a, b, c, x_values)  # y_values.shape = (line_count, x_count)

    # plot_parabola(x_values, y_values)
    return y_values


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()