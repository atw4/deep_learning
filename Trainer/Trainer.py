#!/usr/bin/env python3
import torch

from Utility.Timer import Timer
import Utility.Utility as Utility
from Trainer.TrainerStatistics import TrainerStatistics

class Trainer:
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.max_epochs = max_epochs
        self.num_gpus = num_gpus
        self.gradient_clip_val = gradient_clip_val

        self.gpus = [Utility.gpu(i) for i in range(min(num_gpus, Utility.num_gpus()))]

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()

    def prepare_model(self, model):
        model.to(self.device())
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)

        # Initialize Lazy Parameters
        lazy_batch = self.prepare_batch(next(iter(self.train_dataloader)))
        self.model.training_step(lazy_batch)

        self.optim = model.configure_optimizers()
        self.lr_scheduler = model.configure_lr_scheduler(self.optim)

        num_train_batches = len(self.train_dataloader)
        num_val_batches = (len(self.val_dataloader) if self.val_dataloader is not None else 0)
        self.stats = TrainerStatistics(num_train_batches, num_val_batches)
        
        for _ in range(self.max_epochs):

            self.fit_epoch()

    def prepare_batch(self, batch):
        batch = [a.to(self.device()) for a in batch]

        return batch

    def fit_epoch(self):
        self.model.train()

        self.stats.startEpoch(True)
        epoch_loss = 0
        epoch_accuracy = 0
        for train_batch_idx, batch in enumerate(self.train_dataloader):
            self.stats.startBatch(train_batch_idx)

            loss, accuracy = self.model.training_step(self.prepare_batch(batch))
            scalar_loss = loss.item()
            epoch_loss += scalar_loss
            self.stats.setBatchStat("loss", scalar_loss)

            if accuracy is not None:
                scalar_accuracy = accuracy.item()
                epoch_accuracy += scalar_accuracy
                self.stats.setBatchStat("accuracy", scalar_accuracy)
             
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()

            self.stats.endBatch()
        avg_epoch_loss = epoch_loss / len(self.train_dataloader) if len(self.train_dataloader) > 0 else 0
        avg_epoch_accuracy = epoch_accuracy / len(self.train_dataloader) if len(self.train_dataloader) > 0 else 0
        self.stats.setEpochStat("loss", avg_epoch_loss)
        self.stats.setEpochStat("accuracy", avg_epoch_accuracy)
        self.stats.endEpoch()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()


        if self.val_dataloader is None:
            return
        self.model.eval()

        self.stats.startEpoch(False)
        epoch_loss = 0
        epoch_accuracy = 0
        for val_batch_idx, batch in enumerate(self.val_dataloader):
            self.stats.startBatch(val_batch_idx)

            with torch.no_grad():
                loss, accuracy = self.model.validation_step(self.prepare_batch(batch))
                scalar_loss = loss.item()
                epoch_loss += scalar_loss
                self.stats.setBatchStat("loss", scalar_loss)

                if accuracy is not None:
                    scalar_accuracy = accuracy.item()
                    epoch_accuracy += scalar_accuracy
                    self.stats.setBatchStat("accuracy", scalar_accuracy)


            self.stats.endBatch()

        avg_epoch_loss = epoch_loss / len(self.val_dataloader) if len(self.val_dataloader) > 0 else 0
        avg_epoch_accuracy = epoch_accuracy / len(self.val_dataloader) if len(self.val_dataloader) > 0 else 0
        self.stats.setEpochStat("loss", avg_epoch_loss)
        self.stats.setEpochStat("accuracy", avg_epoch_accuracy)
        self.stats.endEpoch()

    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
        if norm > grad_clip_val:
            for param in params:
                param.grad[:] *= grad_clip_val / norm

    def device(self):
        if self.gpus:
            return self.gpus[0]

        return Utility.cpu()
