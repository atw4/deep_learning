#!/usr/bin/env python3

from Classifer import Classifier
from PatchEmbedding import PatchEmbedding
from ViTBlock import ViTBlock
from torch import nn
import torch

class ViT(Classifier):
    """Vision Transformer."""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens, num_heads, num_blks, emb_dropout, blk_dropout, lr=0.1, use_bias=False, num_classes=10):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_hiddens = num_hiddens
        self.mlp_num_hiddens = mlp_num_hiddens
        self.num_heads = num_heads
        self.num_blks = num_blks
        self.emb_dropout = emb_dropout
        self.blk_dropout = blk_dropout
        self.lr = lr
        self.use_bias = use_bias
        self.num_classes = num_classes

        self.patch_embedding = PatchEmbedding(img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        num_steps = self.patch_embedding.num_patches + 1 #Add the cls token
        #Positional embedings are learnable
        self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, num_hiddens))
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential()
        for i in range(self.num_blks):
            vit_block = ViTBlock(num_hiddens, num_hiddens, mlp_num_hiddens, num_heads, blk_dropout, self.use_bias)
            self.blks.add_module(f"{i}", vit_block)
        self.head = nn.Sequential(nn.LayerNorm(self.num_hiddens), nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])
