#!/usr/bin/env python3

from torch import nn
import torch
from Module import Module
from RNN import RNN

from RNN import RNN 

class BiGRU(RNN):
    def __init__(self, num_inputs, num_hiddens):
        Module.__init__(self)

        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens

        self.rnn = nn.GRU(num_inputs, num_hiddens, bidirectional = True)
        self.num_hiddens *= 2 # The output dimension will be doubled
