#!/usr/bin/env python3

from Utility.Timer import Timer
from ProgressBoard import ProgressBoard
import torch
import time

class TrainerStatistics:
    def __init__(self, capture_train_per_epoch=3, capture_valid_per_epoch=1):
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

    def evaluate_loss(net, data_iter, loss):
        """Evaluate the loss of a model on the given dataset.

        Defined in :numref:`sec_utils`"""
        metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
        for X, y in data_iter:
            out = net(X)
            y = d2l.reshape(y, out.shape)
            l = loss(out, y)
            metric.add(d2l.reduce_sum(l), d2l.size(l))
        return metric[0] / metric[1]

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



        #Calculate the active epoch loss
        self.active_epoch["loss"] = 0
        
        
        epoch_batches = [b for b in self.batches[True] if b["epoch_idx"] == self.active_epoch["epoch_idx"]]
        for batch in epoch_batches:
            if "loss" not in batch:
                pass

            self.active_epoch["loss"] += batch["loss"]

        self.active_epoch["loss"] = self.active_epoch["loss"]/len(epoch_batches)


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
            x = (epoch - 1) + batch_idx/self.num_batches[train]
        else:
            x = epoch

        return x

    def plot(self, x, y, label):
        """Plot a point in animation."""
        self.board.xlabel = 'epoch'

        self.board.draw(x, y.to(torch.device('cpu')).detach().numpy(), label)
