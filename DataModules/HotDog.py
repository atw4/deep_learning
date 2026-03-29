#!/usr/bin/env python3
from DataModules.DataModule import DataModule
import Utility.Utility as Utility
import torchvision
from torchvision import transforms
import os
import torch

class HotDog(DataModule):
    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size

        data_dir = self._download()
        data_dir = os.path.join(data_dir, 'hotdog')

        self.train = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
        self.val = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))


    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=False, num_workers=self.num_of_workers)

    
    def _download(self):
        folder = Utility.extract(Utility.download(Utility.DATA_URL + 'hotdog.zip', sha1_hash='fba480ffa8aa7e0febbb511d181409f899b9baa5'))
        return folder

        
