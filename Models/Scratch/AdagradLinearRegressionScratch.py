
#!/usr/bin/env python3
import torch
from torch import nn
from Models.Module import Module
from Models.Scratch.Adagrad import Adagrad
import math

class AdagradLinearRegressionScratch(Module):
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()

        self.num_inputs = num_inputs
        self.lr = lr
        self.sigma = sigma

        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        return torch.matmul(X, self.w) + self.b


    ### Squared Loss
    def loss(self, y_hat, y):
        #l = ((y_hat - y) ** 2) / 2
        l = ((y_hat - y) ** 2)  #Remove the factor of 2 so it matches the MseLoss implementation
        return l.mean()

    def configure_optimizers(self):
        return Adagrad([self.w, self.b], self.lr)
