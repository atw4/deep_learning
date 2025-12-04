#!/usr/bin/env python3
import torch

class Adadelta():
    def __init__(self, params, rho):
        self.params = params
        self.rho = rho


        self.s = [torch.zeros(p.shape) for p in self.params]
        self.deltas = [torch.zeros(p.shape) for p in self.params]


        self.eps = 1e-6

    def step(self):
        for param, s, delta in zip(self.params, self.s, self.deltas):
            s[:] = self.rho * s + (1 - self.rho)*torch.square(param.grad)
            g = torch.sqrt(delta + self.eps) / torch.sqrt(s + self.eps) * param.grad
            param[:] -= g
            delta[:] = self.rho * delta + (1 - self.rho) * g * g

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
