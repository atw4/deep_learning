#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import typing
from Models.Module import Module

class Classifier(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @torch.jit.export
    def training_step(self, batch: typing.Tuple[torch.Tensor, torch.Tensor]):
        Y_hat = self(*batch[:-1])
        loss = self.loss(Y_hat, batch[-1])
        accuracy = self.accuracy(Y_hat, batch[-1])

        return loss, accuracy

    @torch.jit.export
    def validation_step(self, batch: typing.Tuple[torch.Tensor, torch.Tensor]):
        Y_hat = self(*batch[:-1])
        loss = self.loss(Y_hat, batch[-1])
        accuracy = self.accuracy(Y_hat, batch[-1])

        return loss, accuracy

    def accuracy(self, Y_hat, Y, averaged: bool = True):
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        preds = Y_hat.argmax(dim=1).type(Y.dtype)
        compare = (preds == Y.reshape(-1)).type(torch.float32)

        return compare.mean() if averaged else compare

    def loss(self, Y_hat, Y, averaged: bool = True):
        Y_hat = Y_hat.reshape((-1, Y_hat.shape[-1]))
        Y = Y.reshape((-1,))
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none'
        )

