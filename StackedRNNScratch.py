#!/usr/bin/env python3


from torch import nn
import torch
from Module import Module

from RNNScratch import RNNScratch

class StackedRNNScratch(Module):
    def __init__(self, num_inputs, num_hiddens, num_layers, sigma = 0.01):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.sigma = sigma 

        self.rnns = nn.Sequential(*[RNNScratch(num_inputs if i==0 else num_hiddens, num_hiddens, sigma) for i in range(num_layers)])



    def forward(self, inputs, Hs = None):
        outputs = inputs
        if Hs is None:
            Hs = [None] * self.num_layers

        for i in range(self.num_layers):
            outputs, Hs[i] = self.rnns[i](outputs, Hs[i])
            outputs = torch.stack(outputs, 0)

        return outputs, Hs

