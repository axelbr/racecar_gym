
from torch.optim import Adam
from torch import nn
import torch
from torch.utils.data import DataLoader
from attention_vae import AttentionVAE

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD  = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD


def train():

    # Model Hyperparameters

    model = AttentionVAE

    dataset_path = '~/datasets'

    cuda = True
    DEVICE = torch.device("cuda" if cuda else "cpu")

    #todo: overwrite dims
    batch_size = 100
    x_dim = 784
    hidden_dim = 400
    latent_dim = 200

    lr = 1e-3

    epochs = 30

    kwargs = {'num_workers': 1, 'pin_memory': True}

    # TODO
    train_dataset = None
    test_dataset = None

    # TRAJECTORY format should be [batch_size, num_agents, timesteps, spatial_dim (=2)]
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    BCE_loss = nn.BCELoss()

    optimizer = Adam(model.parameters(), lr=lr)

    print("Start training VAE...")
    model.train()

    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx * batch_size))

    print("Finish!!")