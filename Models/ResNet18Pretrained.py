#!/usr/bin/env python3


import torchvision
import torch
from torch import nn
from Models.Classifer import Classifier

class ResNet18Pretrained(Classifier):
    def __init__(self, lr = 0.1, num_classes = 2):
        super(ResNet18Pretrained, self).__init__()
        self.lr = lr

        self.net = torchvision.models.resnet18(pretrained = True)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)
        nn.init.xavier_uniform_(self.net.fc.weight)
        

    def configure_optimizers(self):
        params_1x = [param for name, param in self.net.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]

        return torch.optim.SGD([{'params': params_1x},
                                   {'params': self.net.fc.parameters(),
                                    'lr': self.lr * 10}],
                                lr=self.lr, weight_decay=0.001)       
