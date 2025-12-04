#!/usr/bin/env python3

from DataModules.DataModule import DataModule
import Utility.Utility as Utility
import numpy as np
import torch
from io import StringIO

class CH11DataModule(DataModule):
    def __init__(self, num_train = 1500, num_val = 0, batch_size = 10):
        self.num_train = num_train
        self.num_val = num_val
        self.batch_size = batch_size

        n = self.num_train + self.num_val
        self.X, self.y = self._preprocess(self._download(), n)

    
    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)
        

    def _preprocess(self, text, n):
        data = np.genfromtxt(StringIO(text),  delimiter='\t')

        data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0)).to(torch.float32)
        data = data[:n]

        X, y = data[:n, :-1], data[:n, -1]
        y = y.unsqueeze(1)

        return X, y

    def _download(self):
        fname = Utility.download(Utility.DATA_URL + 'airfoil_self_noise.dat', sha1_hash='76e5be1548fd8222e5074cf0faae75edff8cf93f')
        with open(fname) as f:
            return f.read()

    
