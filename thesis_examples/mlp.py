import torch
from torch import nn, optim
from torch.nn.modules import Module


class MLP(nn.Module):
    def __init__(self, input_size, output_size, num_layers=2, layers_dim=None):
        super().__init__()

        if layers_dim is None:
            layers_dim = [output_size] * num_layers

        self.layers = nn.ModuleList()
        self.input_size = input_size  # Can be useful later ...
        for size in layers_dim:
            self.layers.append(nn.Linear(input_size, size))
            if size != layers_dim[-1]:
                self.layers.append(torch.nn.ReLU())
            input_size = size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
