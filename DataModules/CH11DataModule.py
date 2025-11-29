#!/usr/bin/env python3

from DataModule import DataModule
import Utility
import numpy as np
import torch
from io import StringIO

class CH11DataModule(DataModule):
    def __init__(self, num_train = 1500, batch_size = 10):
        self.num_train = num_train
        self.batch_size = batch_size

        self.X, self.y = self._preprocess(self._download())

    
    def get_dataloader(self, train):
        return self.get_tensorloader((self.X, self.y), train)
        

    def _preprocess(self, text):
        data = np.genfromtxt(StringIO(text),  delimiter='\t')

        data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0)).to(torch.float32)
        X, y = data[:self.num_train, :-1], data[:self.num_train, -1]
        y = y.unsqueeze(1)

        return X, y

    def _download(self):
        fname = Utility.download(Utility.DATA_URL + 'airfoil_self_noise.dat', sha1_hash='76e5be1548fd8222e5074cf0faae75edff8cf93f')
        with open(fname) as f:
            return f.read()

    
