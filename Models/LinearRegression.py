
#!/usr/bin/env python3
import torch
from torch import nn
from Models.Module import Module


class LinearRegression(Module):
    def __init__(self, lr):
        super().__init__()

        self.lr = lr
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)
