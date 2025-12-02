#!/usr/bin/env python3
import torch
from torch import nn
from Models.Module import Module
from Models.Scratch.SGD import SGD
import math

class LinearRegressionScratch(Module):
    def __init__(self, num_inputs, lr, sigma=0.01, momentum = 0.0):
        super().__init__()

        self.num_inputs = num_inputs
        self.lr = lr
        self.sigma = sigma
        self.momentum = momentum

        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        return torch.matmul(X, self.w) + self.b


    #### Hubers Los

    ### Squared Loss
    def loss(self, y_hat, y):
        l = ((y_hat - y) ** 2)/2
        return l.mean()

    def configure_optimizers(self):
        return SGD([self.w, self.b], self.lr, self.momentum)
