#!/usr/bin/env python3
from DataModule import DataModule
import torch
import random

class DataSample(DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        super().__init__()
        self.num_train = num_train
        self.num_val = num_val
        self.num_inputs = num_inputs
        self.batch_size = batch_size

        n = num_train + num_val

        self.X = torch.randn(n, num_inputs)
        noise = torch.randn(n, 1) * 0.01

        w = torch.ones((num_inputs, 1)) * 0.01
        b = 0.05

        self.y = torch.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)
