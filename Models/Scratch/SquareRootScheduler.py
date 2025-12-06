
#!/usr/bin/env python3
import torch

class SquareRootScheduler():
    def __init__(self, optim):
        self.optim = optim
        print(self.optim)

    def step(self, epoch):
        self.optim.lr = self.optim.lr*pow(epoch + 1.0, -0.5)
