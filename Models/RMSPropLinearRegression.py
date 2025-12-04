#!/usr/bin/env python3
import torch
from torch import nn
from Models.Module import Module


class RMSPropLinearRegression(Module):
    def __init__(self, lr, gamma):
        super().__init__()

        self.lr = lr
        self.gamma = gamma

        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr = self.lr, alpha = self.gamma)
