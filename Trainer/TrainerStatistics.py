#!/usr/bin/env python3

from Utility.Timer import Timer
from ProgressBoard import ProgressBoard
import torch
import time
import math

class TrainerStatistics:
    def __init__(self, num_train_batches, num_val_batches,
                 capture_train_per_epoch=1,
                 capture_valid_per_epoch=1,

                 show_train_epoch_loss_stat=True,
                 show_val_epoch_loss_stat=True,

                 show_train_epoch_accuracy_stat=True,
                 show_val_epoch_accuracy_stat=True):

        self.num_train_batches = num_train_batches
        self.num_val_batches = num_val_batches

        self.show_train_epoch_loss_stat = show_train_epoch_loss_stat
        self.show_val_epoch_loss_stat = show_val_epoch_loss_stat

        self.show_train_epoch_accuracy_stat = show_train_epoch_accuracy_stat
        self.show_val_epoch_accuracy_stat = show_val_epoch_accuracy_stat

        self.train_epochs = [] 
        self.val_epochs = [] 
         
        # Batches train/val
        self.train_batches = []
        self.val_batches = []

        self.train = None

        self.timer = None

        self.board = ProgressBoard()

        self.capture_per_epoch = {
            True : capture_train_per_epoch,
            False : capture_valid_per_epoch
        }

        self.active_batch = None
        self.active_epoch = None

    def startEpoch(self, train):
        self.train = train
        self.active_epoch = {}

        epochs = self.train_epochs if self.train else self.val_epochs
        epochs.append(self.active_epoch)

        self.active_epoch["start_time"] = time.time()


        if len(epochs) <= 1:
            self.active_epoch["rel_start_time"] = 0
            self.active_epoch["epoch_idx"] = 0
        else:
            last_epoch = epochs[-2]
            self.active_epoch["rel_start_time"] = last_epoch["rel_start_time"] + (self.active_epoch["start_time"] - last_epoch["start_time"])
            self.active_epoch["epoch_idx"] = last_epoch["epoch_idx"] + 1

        self.active_epoch["epoch_x"] = self.active_epoch["epoch_idx"] + 1

    def endEpoch(self):
        self.active_epoch["end_time"] = time.time()
        self.active_epoch["rel_end_time"] = self.active_epoch["rel_start_time"] + (self.active_epoch["end_time"] - self.active_epoch["start_time"])
        self.active_epoch["duration"] = self.active_epoch["end_time"] - self.active_epoch["start_time"]


        if self.train:
            if self.show_train_epoch_loss_stat:
                self.plot(self.active_epoch["epoch_x"], self.active_epoch["loss"], "train loss")
            if self.show_train_epoch_accuracy_stat:
                self.plot(self.active_epoch["epoch_x"], self.active_epoch["accuracy"], "train accuracy" )
        else:
            if self.show_val_epoch_loss_stat:
                self.plot(self.active_epoch["epoch_x"], self.active_epoch["loss"], "val loss")
            if self.show_val_epoch_accuracy_stat:
                self.plot(self.active_epoch["epoch_x"], self.active_epoch["accuracy"], "val accuracy")
                
        self.active_epoch = None


    def setEpochStat(self, key, value):
        if self.active_epoch is None:
            return

        self.active_epoch[key] = value
        

            
    def startBatch(self, batch_idx):
        #Should we capture this batch
        num = self.num_train_batches if self.train else self.num_val_batches

        cap = math.ceil(num / self.capture_per_epoch[self.train])

        if batch_idx % cap != 0:
            self.active_batch = None
            return

        epochs = self.train_epochs if self.train else self.val_epochs

        self.active_batch = {"epoch_idx" : self.active_epoch["epoch_idx"], "batch_idx" : batch_idx}
        self.active_batch["epoch_x"] = self.getX(len(epochs), self.active_batch["batch_idx"], self.train)
        self.active_batch["start_time"] = time.time()
        self.active_batch["rel_start_time"] = self.active_epoch["rel_start_time"] + (self.active_batch["start_time"] - self.active_epoch["start_time"])

        if self.train:
            self.train_batches.append(self.active_batch)
        else:
            self.val_batches.append(self.active_batch)

    def setBatchStat(self, key, value):
        if self.active_batch is None:
            return

        self.active_batch[key] = value
            

    def endBatch(self):
        if self.active_batch is None:
            return

        self.active_batch["end_time"] = time.time()
        self.active_batch = None
        

    def getX(self, epoch, batch_idx, train):
        if train:
            x = (epoch - 1) + batch_idx/self.num_train_batches
        else:
            x = epoch

        return x


    def plot(self, x, y, label):
        """Plot a point in animation."""
        self.board.xlabel = 'epoch'

        self.board.draw(x, y, label)

    #Accuracy related stats
    def get_train_epoch_accuracy_stat(self):
        return self.get_stat(self.train_epochs, "epoch_x", "accuracy")
        
    def get_train_batch_accuracy_stat(self):
        return self.get_stat(self.train_batches, "epoch_x", "accuracy")

    def get_val_epoch_accuracy_stat(self):
        return self.get_stat(self.val_epochs, "epoch_x", "accuracy")
        
    def get_val_batch_accuracy_stat(self):
        return self.get_stat(self.val_batches, "epoch_x", "accuracy")

    #Loss related stats
    def get_train_epoch_loss_stat(self):
        return self.get_stat(self.train_epochs, "epoch_x", "loss")
        
    def get_train_batch_loss_stat(self):
        return self.get_stat(self.train_batches, "epoch_x", "loss")

    def get_val_epoch_loss_stat(self):
        return self.get_stat(self.val_epochs, "epoch_x", "loss")
        
    def get_val_batch_loss_stat(self):
        return self.get_stat(self.val_batches, "epoch_x", "loss")
        
    def get_stat(self, array, x_key, y_key):
        return [(e[x_key], e[y_key]) for e in array if x_key in e and y_key in e]


