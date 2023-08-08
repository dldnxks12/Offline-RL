import sys, math, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable


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
        # mask에서 0의 값을 가진 곳을 -1e9 값으로 대체
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


# TODO :  # ------------------------- Enco-Deco Part ------------------------- #

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

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# TODO :  # ------------------------- Batch Part ------------------------- #
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    def __init__(self, src, trg=None, pad = 0):
        # src : batch_size , n_sequence_src
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2) # batch_size, 1, n_sequence_src

        if trg is not None:
            self.trg   = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.mask_std_mask(self.trg, pad) # batch_size, n_seq_trg, n_seq_trg
            self.ntokens = (self.trg_y != pad).data.sum()     # 패딩 토큰이 아닌 토큰 수

    @staticmethod
    def mask_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()

        # size_average and reduce are in the process of being deprecated,
        # and in the meantime, specifying either of those two args will override reduction.
        # self.criterion = nn.KLDivLoss(size_average=False)

        # smoothing을 적용한 타겟과 로스를 구하므로 NLLLoss 대신 KLDivLoss 사용
        self.criterion = nn.KLDivLoss(reduction='sum')  # input: log-probabilities
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        # x: model.generator에서 출력한 log_softmax 값

        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))  # 정답자리와 패딩자리 두자리 빼고
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # 여기까지 스무딩 시켰고...

        # 패딩 토큰 위치는 확률을 0으로 지정
        true_dist[:, self.padding_idx] = 0

        # target이 패팅토큰 번호라면 그 데이터에 대해서는 로스를 구할 필요
        # 없으므로 모든 확률분포자리를 0으로 만들어 버림
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            # index_fill_(dim, index, val): dim차원을 따라 index가 지정된 위치에 val을 채움
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        # loss 계산
        return self.criterion(x, true_dist)

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        # 여기서 generator는 model.generator
        self.generator = generator

        # 여기서 criterion은 LabelSmoothing
        self.criterion = criterion

        self.opt = opt

    def __call__(self, x, y, norm):
        # norm은 batch에서 토큰 수
        # self.ntokens = (self.trg_y != pad).data.sum() # 패딩 토큰이 아닌 토큰 수

        x = self.generator(x)

        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # return loss.data[0] * norm
        return loss * norm


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        # 초기에는 min( , )에서 뒷부분이 작동하여 step에 선형적으로 lr이 증가
        # 그렇게 뒷 부분이 자꾸 커지다 step에 self.warmup과 같아지면
        # 뒷부분이 step*step**(-1.5)=step**(-0.5)가 되고
        # step = self.warmup+1부터는 앞부분이 작아져서
        # 어느 순간 step의 제곱근에 반비례하게 lr이 줄어듬
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))