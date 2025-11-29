#!/usr/bin/env python3
from DataModule import DataModule
import torch
import random

class SyntheticRegressionData(DataModule):
    def __init__(self, w, b, noise = 0.01, num_train = 1000, num_val = 1000, batch_size=32):
        super().__init__()
        self.num_train = num_train
        self.num_val = num_val
        self.batch_size = batch_size

        n = num_train + num_val
        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise
        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)
