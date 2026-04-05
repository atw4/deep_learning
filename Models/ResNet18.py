#!/usr/bin/env python3

import torch
from Models.ResNet import ResNet

class ResNet18(ResNet):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__(((2, 64), (2, 128), (2, 256), (2, 512)), lr, num_classes)

    def configure_optimizers(self):
        return torch.optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=0.001)       
