#!/usr/bin/env python3

from Utility.Timer import Timer
from ProgressBoard import ProgressBoard
import torch
import time

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

        self.active_batch = None
        self.active_epoch = None

    def startEpoch(self):
        self.active_epoch = {}
        self.epochs.append(self.active_epoch)

        self.active_epoch["start_time"] = time.time()


        if len(self.epochs) <= 1:
            self.active_epoch["rel_start_time"] = 0
            self.active_epoch["epoch_idx"] = 0
        else:
            last_epoch = self.epochs[-2]
            self.active_epoch["rel_start_time"] = last_epoch["rel_start_time"] + (self.active_epoch["start_time"] - last_epoch["start_time"])
            self.active_epoch["epoch_idx"] = last_epoch["epoch_idx"] + 1

    def stopEpoch(self):
        self.active_epoch["end_time"] = time.time()
        self.active_epoch["rel_end_time"] = self.active_epoch["rel_start_time"] + (self.active_epoch["end_time"] - self.active_epoch["start_time"])
        self.active_epoch = None

            
    def setNumBatches(self, num_train_batches, num_val_batches):
        self.num_batches[True] = num_train_batches
        self.num_batches[False] = num_val_batches

    def startBatch(self, train, batch_idx):
        self.train = train

        #Should we capture this batch
        cap = self.num_batches[train] // self.capture_per_epoch[train]
        if batch_idx % cap != 0:
            self.active_batch = None
            return

        self.active_batch = {"epoch" : len(self.epochs), "batch_idx" : batch_idx}
        self.active_batch["x"] = self.getX(len(self.epochs), self.active_batch["batch_idx"], train)
        self.active_batch["start_time"] = time.time()
        self.active_batch["rel_start_time"] = self.active_epoch["rel_start_time"] + (self.active_batch["start_time"] - self.active_epoch["start_time"])

        self.batches[train].append(self.active_batch)

    def setBatchStat(self, key, value):
        if self.active_batch is None:
            return

        self.active_batch[key] = value
                
        self.plot(key, value, self.train)

    def endBatch(self):
        if self.active_batch is None:
            return

        self.active_batch["end_time"] = time.time()
        

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
