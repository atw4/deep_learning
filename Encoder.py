#!/usr/bin/env python3

from torch import nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, *args):
        raise NotImplementedError
