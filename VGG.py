#!/usr/bin/env python3

from torch import nn

from Classifer import Classifier

from Utility import vgg_block
from Utility import init_cnn

class VGG(Classifier):
    def __init__(self, arch, lr=0.1, num_classes=10):
        super().__init__()
        self.arch = arch
        self.lr = lr
        self.num_classes = num_classes

        conv_blks = []
        for (num_convs, out_channels) in arch:
            conv_blks.append(vgg_block(num_convs, out_channels))
        self.net = nn.Sequential(
            *conv_blks, nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(num_classes))

        self.net.apply(init_cnn)
