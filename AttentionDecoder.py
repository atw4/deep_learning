#!/usr/bin/env python3

from Decoder import Decoder

class AttentionDecoder(Decoder):
    def __init__(self):
        super().__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError
    
