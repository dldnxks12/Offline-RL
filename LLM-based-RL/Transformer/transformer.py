import sys, math, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from classes import *

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder   = encoder
        self.decoder   = decoder
        self.src_embed = src_embed # 인코더의 임베딩 - 위치인코딩
        self.tgt_embed = tgt_embed # 디코더의 임베딩 - 위치인코딩
        self.generator = generator


    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode( self.encode(src, src_mask), src_mask, tgt, tgt_mask )

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decoder(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

def make_model(src_vocab, tgt_vocab, N = 6, model_dim = 512, d_ff = 2048, h = 8, dropout = 0.1):
    """
    :param src_vocab: 입력을 임베딩할 때 사용하는 단어장 사이즈
    :param tgt_vocab: 출력을 위한 출력 쪽 단어장 사이즈
    :param d_ff: feed_forward dimension
    """

    # TODO : Define modules
    c = copy.deepcopy
    atten    = MultiHeadAttention(h, model_dim)
    ff       = PositionwiseFeedForward(model_dim, d_ff, dropout)
    position = PostionalEncoding(model_dim, dropout)

    # EncoderDecoder(encoder, decoder, src_emb, tgt_emb, generator)
    model    = EncoderDecoder(
        Encoder(EncoderLayer(model_dim, c(atten), c(ff), dropout), N),
        Decoder(DecoderLayer(model_dim, c(atten), c(atten), c(ff), dropout), N),
        nn.Sequential(Embedding(model_dim, src_vocab), c(position)), # Embedding + Positional Encoder 수행해주는 모듈
        nn.Sequential(Embedding(model_dim, tgt_vocab), c(position)), # Embedding + Positional Encoder 수행해주는 모듈
        Generator(model_dim, tgt_vocab)
    )

    # Weight initialize --- 이 부분 매우 Important 하다고 한다.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


if __name__ == "__main__":
    tmp_model = make_model(10, 10, 2)  # Works !