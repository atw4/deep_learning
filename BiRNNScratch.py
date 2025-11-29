#!/usr/bin/env python3

from torch import nn
import torch
from Module import Module
from RNN import RNN

from RNNScratch import RNNScratch 

class BiRNNScratch(Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.sigma = sigma

        self.f_rnn = RNNScratch(num_inputs, num_hiddens, sigma)
        self.b_rnn = RNNScratch(num_inputs, num_hiddens, sigma)
        self.num_hiddens *= 2 # The output dimension will be doubled

    
    def forward(self, inputs, Hs = None):
        f_H, b_H = Hs if Hs is not None else (None, None)
        f_outputs, f_H = self.f_rnn(inputs, f_H)
        b_outputs, b_H = self.b_rnn(reversed(inputs), b_H)
        outputs = [torch.cat((f, b), -1) for f, b in zip(f_outputs, reversed(b_outputs))]

        return outputs, (f_H, b_H)
