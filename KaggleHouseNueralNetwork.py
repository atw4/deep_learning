#!/usr/bin/env python3


import torch
from torch import nn
from Module import Module
from SGD import SGD

class KaggleHouseNueralNetwork(Module):
    def __init__(self, num_hiddens, wd, lr):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.wd = wd
        self.lr = lr

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_hiddens),
            nn.ReLU(),
            nn.LazyLinear(1)
        )

    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net[1].weight, 'weight_decay': self.wd}

        ], lr=self.lr)
