#!/usr/bin/env python3
from Classifer import Classifier
from torch import nn

class DropoutMLP(Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2, dropout_1, dropout_2, lr):
        super().__init__()

        self.num_outputs = num_outputs
        self.num_hiddens_1 =  num_hiddens_1
        self.num_hiddens_2 =  num_hiddens_2
        self.dropout_1 =  dropout_1
        self.dropout_2 =  dropout_2
        self.lr =  lr

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(num_hiddens_1),
            nn.ReLU(),
            nn.Dropout(dropout_1),
            nn.LazyLinear(num_hiddens_2),
            nn.ReLU(),
            nn.Dropout(dropout_2),
            nn.LazyLinear(num_outputs)
        )
