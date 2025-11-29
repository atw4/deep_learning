#!/usr/bin/env python3

from torch import nn
from Classifer import Classifier
from ResNeXtBlock import ResNeXtBlock
import Utility

class AnyNet(Classifier):
    def __init__(self, arch, stem_channels, lr=0.1, num_classes=10):
        super(AnyNet, self).__init__()
        self.arch = arch
        self.stem_channels = stem_channels
        self.lr = lr
        self.num_classes = num_classes

        for i, s in enumerate(arch):
            self.net.add_module(f'stage{i+1}', self.stage(*s))
        self.net.add_module('head', nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.LazyLinear(num_classes)))

        self.net.apply(Utility.init_cnn)

    def stem(self, num_channels):
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=3, stride=2, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLU()
        )

    def stage(self, depth, num_channels, groups, bot_mul):
        blk = []
        for i in range(depth):
            if i == 0:
                blk.append(ResNeXtBlock(num_channels, groups, bot_mul, use_1x1conv=True, strides=2))
            else:
                blk.append(ResNeXtBlock(num_channels, groups, bot_mul))

        return nn.Sequential(*blk)
