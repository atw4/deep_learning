#!/usr/bin/env python3
import torch

class Adagrad():
    def __init__(self, params, lr):
        self.params = params

        self.s = [torch.zeros(p.shape) for p in self.params]

        self.lr = lr
        self.eps = 1e-6

    def step(self):
        for param, s in zip(self.params, self.s):
            s += torch.square(param.grad)
            param -= self.lr * param.grad / torch.sqrt(s + self.eps)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
