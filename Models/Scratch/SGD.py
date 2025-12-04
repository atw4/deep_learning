#!/usr/bin/env python3
import torch

class SGD():
    def __init__(self, params, lr, momentum = 0.0):
        self.params = params
        self.vs = [torch.zeros(p.shape) for p in self.params]

        self.lr = lr
        self.momentum = momentum

    def step(self):
        for param, v in zip(self.params, self.vs):
            v[:] = self.momentum * v + param.grad
            param[:] -= self.lr * v

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
