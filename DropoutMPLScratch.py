#!/usr/bin/env python3
from Classifer import Classifier
from torch import nn

from Utility import dropout_layer

class DropoutMLPScratch(Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2, dropout_1, dropout_2, lr):
        super().__init__()

        self.num_outputs = num_outputs
        self.num_hiddens_1 =  num_hiddens_1
        self.num_hiddens_2 =  num_hiddens_2
        self.dropout_1 =  dropout_1
        self.dropout_2 =  dropout_2
        self.lr =  lr

        self.lin1 = nn.LazyLinear(num_hiddens_1)
        self.lin2 = nn.LazyLinear(num_hiddens_1)
        self.lin3 = nn.LazyLinear(num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((X.shape[0], -1))))
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = self.relu(self.lin2(H1))
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)

        return self.lin3(H2)
