#!/usr/bin/env python3

from LinearRegression import LinearRegression
import torch


class WeightDecay(LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)

        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}], lr=self.lr)
