#!/usr/bin/env python3

from torch import nn

import Utility
from DenseBlock import DenseBlock
from Classifer import Classifier

class DenseNet(Classifier):
    def __init__(self, num_channels=64, growth_rate=32, arch=(4, 4, 4, 4), lr=0.1, num_classes=10):
        super(DenseNet, self).__init__()
        self.num_channels = num_channels
        self.growth_rate = growth_rate
        self.arch = arch
        self.lr = lr
        self.num_classes = num_classes

        self.net = nn.Sequential(self.b1())
        for i, num_convs in enumerate(arch):
            self.net.add_module(f'dense_blk{i+1}', DenseBlock(num_convs, growth_rate))

            # The number of output channels in the previous dense block
            num_channels += num_convs * growth_rate
            # A transition layer that halves the number of channels is added
            # between the dense blocks
            if i != len(arch) - 1:
                num_channels //= 2
                self.net.add_module(f'tran_blk{i+1}', Utility.transition_block(num_channels))

        self.net.add_module('last', nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(num_classes)))

        self.net.apply(Utility.init_cnn)


    def b1(self):
        return nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
