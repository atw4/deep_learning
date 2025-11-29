#!/usr/bin/env python3


from torch import nn

from Classifer import Classifier

from Utility import nin_block
from Utility import init_cnn

class NiN(Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nin_block(96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nin_block(384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            nin_block(num_classes, kernel_size=3, strides=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())


        self.net.apply(init_cnn)
