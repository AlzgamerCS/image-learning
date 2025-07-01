import torch
from torch import nn
import matplotlib.pyplot as plt


def train(model, X, Y, epochs, lr, device):  # returns losses for each epoch
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return losses


def plot_loss(losses):  # plots the loss curve
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.show()
