#!/usr/bin/env python3
from torch import nn

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.mods = []

        for idx, module in enumerate(args):
            self.mods.append(module)

    def forward(self, X):
        for module in self.mods:
            X = module(X)
        return X
