#!/usr/bin/env python3

from Utility.Timer import Timer
from ProgressBoard import ProgressBoard
import torch
import time
import math

class TrainerStatistics:
    def __init__(self, num_train_batches, num_val_batches, capture_train_per_epoch=3, capture_valid_per_epoch=1):
        self.num_train_batches = num_train_batches
        self.num_val_batches = num_val_batches
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

        self.active_epoch["epoch_x"] = self.active_epoch["epoch_idx"] + 1

    def endEpoch(self):
        self.active_epoch["end_time"] = time.time()
        self.active_epoch["rel_end_time"] = self.active_epoch["rel_start_time"] + (self.active_epoch["end_time"] - self.active_epoch["start_time"])
        self.active_epoch["duration"] = self.active_epoch["end_time"] - self.active_epoch["start_time"]

        self.active_epoch = None

    def setEpochStat(self, key, value):
        if self.active_epoch is None:
            return

        self.active_epoch[key] = value
        

            
    def startBatch(self, train, batch_idx):
        self.train = train

        #Should we capture this batch
        num = self.num_train_batches if train else self.num_val_batches

        cap = math.ceil(num / self.capture_per_epoch[train])

        if batch_idx % cap != 0:
            self.active_batch = None
            return

        self.active_batch = {"epoch_idx" : self.active_epoch["epoch_idx"], "batch_idx" : batch_idx}
        self.active_batch["epoch_x"] = self.getX(len(self.epochs), self.active_batch["batch_idx"], train)
        self.active_batch["start_time"] = time.time()
        self.active_batch["rel_start_time"] = self.active_epoch["rel_start_time"] + (self.active_batch["start_time"] - self.active_epoch["start_time"])

        self.batches[train].append(self.active_batch)

    def setBatchStat(self, key, value):
        if self.active_batch is None:
            return

        self.active_batch[key] = value
            



    def endBatch(self):
        if self.active_batch is None:
            return

        self.active_batch["end_time"] = time.time()
        

    def getX(self, epoch, batch_idx, train):
        if train:
            x = (epoch - 1) + batch_idx/self.num_train_batches
        else:
            x = epoch

        return x

    def plot(self, x, y, label):
        """Plot a point in animation."""
        self.board.xlabel = 'epoch'

        self.board.draw(x, y.to(torch.device('cpu')).detach().numpy(), label)

    def get_stat(self, statType, x_key, y_key):
        ret = []

        ret = []
        if statType == "epoch":
            ret = [(e[x_key], e[y_key]) for e in self.epochs if x_key in e and y_key in e]
        elif statType == "train_batch":
            ret = [(e[x_key], e[y_key]) for e in self.batches[True] if x_key in e and y_key in e]

        return ret
                
            
