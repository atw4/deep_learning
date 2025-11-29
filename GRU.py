#!/usr/bin/env python3

from torch import nn
import torch
from Module import Module
from RNN import RNN

class GRU(RNN):
    def __init__(self, num_inputs, num_hiddens, num_layers, dropout = 0):
        Module.__init__(self)
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.dropout = dropout

        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers, dropout = dropout)
