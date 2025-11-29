#!/usr/bin/env python3
from torch import nn
import torch

import torch.nn.functional as F

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.rand_weight = torch.rand((20, 20))
        self.linear = nn.LazyLinear(20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(X @ self.rand_weight + 1)

        X = self.linear(X)

        while X.abs().sum() > 1:
            X /= 2

        return X.sum()
