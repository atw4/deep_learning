#!/usr/bin/env python3
import torch

class DataModule():
    def __init__(self, root="./data", num_of_workers = 2):
        self.root = root
        self.num_of_workers = num_of_workers

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        dataset = torch.utils.data.TensorDataset(*tensors)

        return torch.utils.data.DataLoader(dataset, self.batch_size, shuffle=train)


    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train = True)

    def val_dataloader(self):
        return self.get_dataloader(train = False)
