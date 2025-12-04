#!/usr/bin/env python3
import torch

class Adam():
    def __init__(self, params, lr, beta1, beta2, t = 1):
        self.params = params

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2

        self.t = t


        self.s = [torch.zeros(p.shape) for p in self.params]
        self.v = [torch.zeros(p.shape) for p in self.params]


        self.eps = 1e-6

    def step(self):
        for param, v, s in zip(self.params, self.v, self.s):
            v[:] = self.beta1 * v + (1 - self.beta1)*param.grad
            s[:] = self.beta2 * s + (1 - self.beta2)*torch.square(param.grad)
            v_bias_corr = v/(1 - self.beta1**self.t)
            s_bias_corr = s/(1 - self.beta2**self.t)
            param[:] -= (self.lr*v_bias_corr)/(torch.sqrt(s_bias_corr) + self.eps)

        self.t += 1


    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
