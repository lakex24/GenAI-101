import logging
import argparse

import numpy as np

import torch
# import torch.nn as nn

import generators.tranformer as tf

import data_utils.generate_parabola as gp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)


def run_training():

    data = torch.tensor(gen_parabola_data(), dtype=torch.float32)

    model = tf.MiniTransformer(
        dim=10, heads=2, depth=2, seq_length=10, num_tokens=10
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(data.unsqueeze(0))  # Assume autoregressive generation
        loss = loss_fn(output, data.unsqueeze(0))
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss {loss.item()}")



def gen_parabola_data():
    line_count = 20
    a, b, c = gp.generate_parabola(0, 0, line_count)  # a.shape = (line_count,), same for b, c

    x_count = 10
    x_values = np.linspace(0, x_count, x_count + 1)  # x_values.shape = (x_count + 1,)

    y_values = gp.get_parabola_y_values(a, b, c, x_values)  # y_values.shape = (line_count, x_count + 1)

    # plot_parabola(x_values, y_values)
    return y_values


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()