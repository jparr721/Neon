import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.autograd
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from unet import UNet, initialize_weights
from unetdataset import UNetDataset

dataset = UNetDataset()
dataset.load()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_network(*, batch_size: int, epochs: int, expo: int, lr: float):
    network = UNet(dataset.shape[1], channel_exponent=expo)
    network.to(device)
    print(f"Trainable parameters: {network.n_parameters}")
    network.apply(initialize_weights)
    L1_norm = nn.L1Loss()
    optimizerG = optim.Adam(network.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0)

    dataset.current_input_data.to(device)
    dataset.current_target_data.to(device)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    dataset.set_current(train=False)
    dataset.current_input_data.to(device)
    dataset.current_target_data.to(device)
    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    targets = torch.autograd.Variable(
        torch.cuda.FloatTensor(batch_size, dataset.shape[1], dataset.shape[2], dataset.shape[3]))
    inputs = torch.autograd.Variable(
        torch.cuda.FloatTensor(batch_size, dataset.shape[1], dataset.shape[2], dataset.shape[3]))

    history_L1 = []

    # For validation data
    history_L1_val = []

    if os.path.isfile("models/network"):
        print("Found network, skipping training cycle")
        network.load_state_dict(torch.load("models/network"))
    else:
        print("Training network")
        for epoch in range(epochs):
            network.train()
            L1_accum = 0.0

            for i, (current_input, current_target) in enumerate(train_dataloader, 0):
                inputs.data.copy_(current_input.float())
                targets.data.copy_(current_target.float())

                network.zero_grad()
                prediction = network(inputs)

                lossL1 = L1_norm(prediction, targets)
                lossL1.backward()
                optimizerG.step()
                L1_accum += lossL1.item()

            history_L1.append(L1_accum / len(train_dataloader))

            if epoch < 3 or epoch % 20 == 0:
                print("Epoch: {}, L1 train: {:7.5f}".format(epoch, history_L1[-1]))

        Path("models").mkdir(parents=True, exist_ok=True)
        torch.save(network.state_dict(), "models/network")
        print("Training done, saved network")

    l1train = np.asarray(history_L1)

    plt.plot(np.arange(l1train.shape[0]), l1train, 'b', label='Training loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_network(batch_size=10, epochs=100, expo=3, lr=0.00002)
