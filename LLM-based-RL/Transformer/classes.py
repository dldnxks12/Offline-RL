import sys, math, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# TODO :  # ------------------------- Embedding Part ------------------------- #
# TODO : Word Embedding
class Embedding(nn.Module):
    def __init__(self, model_dim, vocab):
        super(Embedding, self).__init__()
        self.lut       = nn.Embedding(vocab, model_dim)
        self.model_dim = model_dim

    def forward(self, x):
        # postional vector 가 더해지면서 임베딩 벡터 값이 희석되는 걸 방지하기 위해
        # 뒤에 값을 곱해줌 - trick
        return self.lut(x) * math.sqrt(self.model_dim)


# TODO : Postional Encoding
class PostionalEncoding(nn.Module):
    def __init__(self, model_dim, dropout_ratio, max_len = 5000):
        super(PostionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout_ratio)

        pe           = torch.zeros(max_len, model_dim)
        position     = torch.arange(0, max_len).unsqueeze(1)
        div_term     = torch.exp(torch.arange(0, model_dim, 2) * -(math.log(10000.0) / model_dim))
        pe[:, 0::2]  = torch.sin(position * div_term)
        pe[:, 1::2]  = torch.cos(position * div_term)
        pe           = pe.unsqueeze(0)

        self.register_buffer('pe', pe) # optimizer 가 연산하지 않지만, state_dict 로 확인이 가능

    def forward(self, x):
        pe_val = self.pe[:, :x.size(1)]
        pe_val.requires_grad = False
        x = x + pe_val         # Embedding 된 word vector 에 posiotional info 희석해주기
        return self.dropout(x) # Add regularization effect


# TODO :  # ------------------------- Attention Part ------------------------- #

# TODO : Define attention module : Scaled-dot product attention!!
# TODO : this function will be used inside of Multihead-Attention class
def attention(query, key, value, mask = None, dropout = None):
    """
    :param query: (batch_size, h, sequence_num, dimension) # h : num-head , 6 in paper
    :param key  : (batch_size, h, sequence_num, dimension)
    :param value: (batch_size, h, sequence_num, dimension)
    :return     : softmax(QK/root_dim)*V, softmax(QK)
    """
    d_k = query.size(-1)

    # QK
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attention = F.softmax(scores, dim = -1)

    if dropout is not None:
        p_attention = dropout(p_attention)

    return torch.matmul(p_attention, value), p_attention


# TODO : We will stack Attention module x 6 times in Encoder / Decoder block
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):
    def __init__(self, h, model_dim, dropout_ratio = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert model_dim % h == 0

        self.d_k = model_dim // h
        self.h   = h

        self.linears = clones(nn.Linear(model_dim, model_dim), 4)
        self.atten   = None
        self.dropout = nn.Dropout(p = dropout_ratio)

    def forward(self, query, key, value, mask = None):
        batch_size = query.size(0)

        if mask is not None:
            mask = mask.unsqueeze(1)

        query, key, value = [layer(x).view(batch_size, -1, self.h, self.d_k).transpose(1,2) \
                             for layer, x in zip(self.linears, (query, key, value)) ]

        x, self.atten = attention(query, key, value, mask=mask, dropout=self.dropout)

        # Concatenate all sub-attentions
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        # forward of multi-head attention
        return self.linears[-1](x)

# TODO :  # ------------------------- Noramlize Part ------------------------- #

class SublayerConnection(nn.Module):
    # Residual connection
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm    = nn.LayerNorm(size)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, sublayer):
        # apply residual connection to any sublayer with the same size
        # if dropout is None
        # return self.norm(x + sublayer(x))
        return self.norm(x + self.dropout(sublayer(x)))

# TODO :  # ------------------------- Encoder Part ------------------------- #

class EncoderLayer(nn.Module):
    def __init__(self, size, self_atten, feed_forward, dropout):
        super(EncoderLayer, self).__init__()

        self.self_atten   = self_atten   # TODO : MultiHeadAttention
        self.feed_forward = feed_forward # TODO : forward

        # SublayerConnection 2개
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size     = size # model_dim

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x : self.self_atten(x, x, x, mask)) # Q, K, V all same
        x = self.sublayer[1](x, self.feed_forward)
        return x

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N) # layer : EncoderLayer

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# TODO :  # ------------------------- Decoder Part ------------------------- #

class DecoderLayer(nn.Module):
    def __init__(self, size, self_atten, src_atten, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size  # model_dim
        self.self_atten   = self_atten # TODO : MultiHeadAttention in Decoder side
        self.src_atten    = src_atten  # TODO : MultiHeadAttention from Encoder side
        self.feed_forward = feed_forward
        self.sublayer     = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # TODO : memory - encoder로 부터 넘어온 정보
        # TODO : decoder 는 encoder와 다르게 3개의 층
        m = memory
        x = self.sublayer[0](x, lambda x : self.self_atten(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x : self.src_atten(x, m, m, src_mask)) # Cross attention
        x = self.sublayer[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return x


# TODO :  # ------------------------- Final Output Part ------------------------- #

class Generator(nn.Module):
    # (sequenc_num, model_dim) -> (sequenc_num, vocab)
    def __init__(self, model_dim, vocab):
        super(Generator, self).__init__()
        self.projection = nn.Linear(model_dim, vocab)

    def forward(self, x):
        return F.log_softmax(self.projection(x), dim = -1)