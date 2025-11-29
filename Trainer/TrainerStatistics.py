#!/usr/bin/env python3

from Utility.Timer import Timer
from ProgressBoard import ProgressBoard
import torch

class TrainerStatistics:
    def __init__(self, capture_train_per_epoch=1, capture_valid_per_epoch=1):
        self.epochs = [] 
         
        # Batches train/val
        self.batches = {
            True : [],
            False : []
        }

        self.train = None

        self.timer = None

        self.board = ProgressBoard()

        self.capture_per_epoch = {
            True : capture_train_per_epoch,
            False : capture_valid_per_epoch
        }

        self.num_batches = {
            True : 0,
            False : 0
        }

        self.activeBatch = None

    def startEpoch(self):
        self.timer = Timer()
        self.epochs.append({})

    def stopEpoch(self):
        self.timer.stop()
        self.epochs[-1]["time"] = self.timer.cumsum()
        self.timer = None

    def setNumBatches(self, num_train_batches, num_val_batches):
        self.num_batches[True] = num_train_batches
        self.num_batches[False] = num_val_batches


    def startBatch(self, train, batch_idx):
        self.train = train

        #Should we capture this batch
        cap = self.num_batches[train] // self.capture_per_epoch[train]
        if batch_idx % cap != 0:
            self.activeBatch = None
            return

        self.activeBatch = {"epoch" : len(self.epochs), "batch_idx" : batch_idx}
        self.activeBatch["x"] = self.getX(self.activeBatch["epoch"], self.activeBatch["batch_idx"], train)

        self.batches[train].append(self.activeBatch)

    def setBatchStat(self, key, value):
        if self.activeBatch is None:
            return

        self.activeBatch[key] = value
                
        self.plot(key, value, self.train)

    def getX(self, epoch, batch_idx, train):
        if train:
            x = (epoch - 1) + batch_idx/self.num_batches[train]
        else:
            x = epoch

        return x
            
    def plot(self, key, value, train):
        """Plot a point in animation."""
        self.board.xlabel = 'epoch'

        x = self.batches[train][-1]["x"]

        self.board.draw(x, value.to(torch.device('cpu')).detach().numpy(), ('train_' if train else 'val_') + key)
