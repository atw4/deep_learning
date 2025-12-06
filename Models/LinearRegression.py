
#!/usr/bin/env python3
import torch
from torch import nn
from Models.Module import Module


class LinearRegression(Module):
    def __init__(self, optimizer, **kwargs):
        super().__init__(**kwargs)

        self.optimizer = optimizer
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y), None

    def configure_optimizers(self):
        return self.optimizer(self.parameters())
