#!/usr/bin/env python3

from LinearRegressionScratch import LinearRegressionScratch


class WeightDecayScratch(LinearRegressionScratch):
    def __init__(self, num_inputs, lambd, lr, sigma=0.01):
        super().__init__(num_inputs, lr, sigma)

        self.lambd = lambd

    def l2_penalty(self, w):
        return (w ** 2).sum() /2

    def loss(self, y_hat, y):
        return (super().loss(y_hat, y)) + self.lambd * self.l2_penalty(self.w)
