#!/usr/bin/env python3
import torch
from torch import nn

from ProgressBoard import ProgressBoard

class Module(nn.Module):
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=2):
        super().__init__()
        self.board = ProgressBoard()
        self.plot_train_per_epoch=plot_train_per_epoch
        self.plot_valid_per_epoch=plot_valid_per_epoch

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is not defined'
        return self.net(X)

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)


    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        self.board.draw(x, value.to(torch.device('cpu')).detach().numpy(),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def apply_init(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)
