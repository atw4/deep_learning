#!/usr/bin/env python3

from torch import nn

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, *args):
        raise NotImplementedError
