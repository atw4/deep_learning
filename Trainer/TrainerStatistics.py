#!/usr/bin/env python3

from Utility.Timer import Timer
from ProgressBoard import ProgressBoard
import torch

class TrainerStatistics:
    def __init__(self, plot_train_per_epoch=1, plot_valid_per_epoch=1):
        self.epochs = [] 
        self.train_batches = []
        self.val_batches = []
        self.train = None

        self.timer = None

        self.board = ProgressBoard()
        self.plot_train_per_epoch=plot_train_per_epoch
        self.plot_valid_per_epoch=plot_valid_per_epoch

    def startEpoch(self, num_train_batches, num_val_batches):
        self.timer = Timer()
        self.epochs.append(
            {'stats' : {},
             'num_train_batches' : num_train_batches,
             'num_val_batches' : num_val_batches
            })

    def stopEpoch(self):
        self.timer.stop()
        self.epochs[-1]["time"] = self.timer.cumsum()
        self.timer = None

    def startBatch(self, train):
        self.train = train

        if self.train:
            self.train_batches.append({"epoch" : len(self.epochs), "stats" : {}})
        else:
            self.val_batches.append({"epoch" : len(self.epochs), "stats" : {}})

    def setBatchStat(self, key, value):
        if self.train:
            self.train_batches[-1]["stats"][key] = value
        else:
            self.train_batches[-1]["stats"][key] = value
                
        self.plot(key, value, self.train)
        

    def plot(self, key, value, train):
        """Plot a point in animation."""
        self.board.xlabel = 'epoch'
        epoch = len(self.epochs)
        if train:
            train_batch_idx = len(self.train_batches)
            num_train_batches = self.epochs[-1]["num_train_batches"]


            x = train_batch_idx / num_train_batches
            n = num_train_batches / self.plot_train_per_epoch
        else:
            num_val_batches = self.epochs[-1]["num_val_batches"]

            x = epoch + 1
            n = num_val_batches / self.plot_valid_per_epoch


        self.board.draw(epoch + 1, value.to(torch.device('cpu')).detach().numpy(),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))
