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
        self.stats = TrainerStatistics()

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()

    def prepare_model(self, model):
        model.trainer = self

        model.to(self.device())

        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()

        num_train_batches = len(self.train_dataloader)
        num_val_batches = (len(self.val_dataloader) if self.val_dataloader is not None else 0)
        self.stats.setNumBatches(num_train_batches, num_val_batches)
        
        for _ in range(self.max_epochs):

            self.stats.startEpoch()
            self.fit_epoch()
            self.stats.stopEpoch()

    def prepare_batch(self, batch):
        batch = [a.to(self.device()) for a in batch]

        return batch

    def fit_epoch(self):
        self.model.train()

        for train_batch_idx, batch in enumerate(self.train_dataloader):
            self.stats.startBatch(True, train_batch_idx)

            loss = self.model.training_step(self.prepare_batch(batch))
            self.stats.setBatchStat("loss", loss)
             
            self.optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                if self.gradient_clip_val > 0:
                    self.clip_gradients(self.gradient_clip_val, self.model)
                self.optim.step()

            self.stats.endBatch()


        if self.val_dataloader is None:
            return
        self.model.eval()

        for val_batch_idx, batch in enumerate(self.val_dataloader):
            self.stats.startBatch(False, val_batch_idx)

            with torch.no_grad():
                loss = self.model.validation_step(self.prepare_batch(batch))
                self.stats.setBatchStat("loss", loss)

            self.stats.endBatch()

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
