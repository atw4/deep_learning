#!/usr/bin/env python3
from DataModules.DataModule import DataModule
import Utility.Utility as Utility
import torchvision
from torchvision import transforms
import os
import torch

class HotDog(DataModule):
    def __init__(self, batch_size=16):
        super().__init__(num_of_workers=0)
        self.batch_size = batch_size

        data_dir = self._download()
        data_dir = os.path.join(data_dir, 'hotdog')

        # Specify the means and standard deviations of the three RGB channels to
        # standardize each channel
        normalize = torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.train_augs = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize])

        self.val_augs = torchvision.transforms.Compose([
            torchvision.transforms.Resize([256, 256]),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize])

        self.train_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=self.train_augs)
        self.val_dataset = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=self.val_augs)


    def get_dataloader(self, train):
        if train:
            return torch.utils.data.DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_of_workers)
        else:
            return torch.utils.data.DataLoader(self.val_dataset, self.batch_size, shuffle=False, num_workers=self.num_of_workers)
            
    
    def _download(self):
        folder = Utility.extract(Utility.download(Utility.DATA_URL + 'hotdog.zip', sha1_hash='fba480ffa8aa7e0febbb511d181409f899b9baa5'))
        return folder

        
