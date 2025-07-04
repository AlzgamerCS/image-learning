import torch.nn as nn
import torch.nn.functional as F


class ImageMLP(nn.Module):
    def __init__(self, input_size = 2, width = 64, depth = 3, output_size = 1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_size, width))
        layers.append(nn.ReLU())

        for _ in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(width, output_size))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
    def forward(self, x):
        
        x = self.model(x)

        return x
