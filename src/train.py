import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


def train(model, X, Y, epochs, lr, device, batch_size=1024):
    """
    Батчевый градиентный спуск.
    Args:
        model: ваш ImageMLP
        X: torch.Tensor, shape [N, 2]
        Y: torch.Tensor, shape [N, C]
        epochs: int
        lr: float
        device: torch.device
        batch_size: int — размер батча
    Returns:
        losses: list[float] — средний loss по батчам каждой эпохи
    """
    # Датасет и лоадер
    dataset = TensorDataset(X, Y)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    losses = []
    for epoch in range(1, epochs+1):
        running_loss = 0.0
        batches = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches += 1

        avg_loss = running_loss / batches
        losses.append(avg_loss)
        print(f"Epoch {epoch:3d}/{epochs} — Loss: {avg_loss:.6f}")

    return losses
def plot_loss(losses):  # plots the loss curve
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.show()
