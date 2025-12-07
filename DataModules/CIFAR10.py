#!/usr/bin/env python3
from DataModules.DataModule import DataModule
import torchvision
from torchvision import transforms
import torch

class CIFAR10(DataModule):
    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size

        train_aug = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()])

        val_aug = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])


        self.train = torchvision.datasets.CIFAR10(
            root = self.root,
            train = True,
            download = True,
            transform=train_aug)
        self.val = torchvision.datasets.CIFAR10(
            root = self.root,
            train = False,
            download = True,
            transform=val_aug)

    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=False, num_workers=self.num_of_workers)
