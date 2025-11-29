#!/usr/bin/env python3
from Classifer import Classifier
from torch import nn

class SoftmaxRegression(Classifier):
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.num_outputs = num_outputs
        self.lr = lr

        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_outputs))


    def forward(self, X):
        return self.net(X)
