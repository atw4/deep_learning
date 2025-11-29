#!/usr/bin/env python3
from RNN import RNN
from Module import Module
from torch import nn

class LSTM(RNN):
    def __init__(self, num_inputs, num_hiddens):
        Module.__init__(self)
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.rnn = nn.LSTM(num_inputs, num_hiddens)

    def forward(self, inputs, H_C = None):
        return self.rnn(inputs, H_C)
        
    
    

    
