#!/usr/bin/env python3
from Encoder import Encoder
from PositionalEncoding import PositionalEncoding
from TransformerEncoderBlock import TransformerEncoderBlock
from torch import nn

import math

class TransformerEncoder(Encoder):
    """The Transformer encoder"""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            encoder_blk = TransformerEncoderBlock(num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias)
            self.blks.add_module("block" + str(i), encoder_blk)

    def forward(self, X, valid_lens):
        # Since positional encoding values are between -1 and 1,
        # the embedding values are multipled by the square root of the embedding dimension
        # to rescale before ther are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X
            
