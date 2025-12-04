#!/usr/bin/env python3
import torch
from torch import nn


class Module(nn.Module):
    def __init__(self):
        super().__init__()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is not defined'
        return self.net(X)

    def training_step(self, batch):
        Y_hat = self(*batch[:-1])
        x = batch[-1]
        l = self.loss(Y_hat, x)

        return l

    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        x = batch[-1]
        l = self.loss(Y_hat, x)

        return l


    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def configure_lr_scheduler(self, optim):
        return None

    def apply_init(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)
