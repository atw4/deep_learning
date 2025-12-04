#!/usr/bin/env python3
import torch
from torch import nn
from Models.Module import Module
from Models.Scratch.SGD import SGD
import math

class LinearRegressionScratch(Module):
    def __init__(self, num_inputs, optimizer, sigma=0.01):
        super().__init__()

        self.num_inputs = num_inputs
        self.optimizer = optimizer

        self.sigma = sigma

        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        return torch.matmul(X, self.w) + self.b


    #### Hubers Los

    ### Squared Loss
    def loss(self, y_hat, y):
        #ll = ((y_hat - y) ** 2) / 2
        l = ((y_hat - y) ** 2)  #Remove the factor of 2 so it matches the MseLoss implementation
        return l.mean()

    def configure_optimizers(self):
        return self.optimizer([self.w, self.b])
