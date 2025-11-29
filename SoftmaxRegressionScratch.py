#!/usr/bin/env python3
import torch
from Classifer import Classifier
from Utility import softmax, cross_entropy

class SoftmaxRegressionScratch(Classifier):
    def  __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.lr = lr
        self.sigma = sigma

        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs), requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]

    def forward(self, X):
        X = X.reshape((-1, self.W.shape[0]))
        return softmax(torch.matmul(X, self.W) + self.b)

    def loss(self, y_hat, y):
        return cross_entropy(y_hat, y)
