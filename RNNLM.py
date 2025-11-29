#!/usr/bin/env python3

from torch import nn
from RNNLMScratch import RNNLMScratch

class RNNLM(RNNLMScratch):
    def init_params(self):
        self.linear = nn.LazyLinear(self.vocab_size)

    def output_layer(self, hiddens):
        return self.linear(hiddens).swapaxes(0, 1)
