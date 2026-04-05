#!/usr/bin/env python3


import torchvision
from torch import nn
from Models.Classifer import Classifier

class ResNet18Pretrained(Classifier):
    def __init__(self, lr = 0.1, num_output = 2):
        super(ResNet18Pretrained, self).__init__(lr)

        self.net = torchvision.models.resnet18(pretrained = True)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_output)
        nn.init_xavier_uiniform_(self.net.fc.weight)
        
