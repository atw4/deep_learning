#!/usr/bin/env python3
import torch
from torch import nn
import typing


class Module(nn.Module):
    def __init__(self, lr_scheduler=None, **kwargs):
        super().__init__(**kwargs)
        self.lr_scheduler = lr_scheduler

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is not defined'
        return self.net(X)

    @torch.jit.export
    def training_step(self, batch: typing.Tuple[torch.Tensor, torch.Tensor]):
        Y_hat = self(*batch[:-1])
        x = batch[-1]
        l = self.loss(Y_hat, x)

        return l

    @torch.jit.export
    def validation_step(self, batch: typing.Tuple[torch.Tensor, torch.Tensor]):
        Y_hat = self(*batch[:-1])
        x = batch[-1]
        l = self.loss(Y_hat, x)

        return l


    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def configure_lr_scheduler(self, optim):
        if self.lr_scheduler is None:
            return None

        return self.lr_scheduler(optim)

    def apply_init(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)

    def layer_summary(self, X_shape):
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)
