import sys, math, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from classes import *

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


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss   = 0
    tokens       = 0

    # data_iter를 만들 때 설정한 nbatches 만큼 루프를 돈다. 즉 1에폭
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)

        # 여기서 loss_compute()는 SimpleLossCompute 임
        loss          = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss   += loss
        total_tokens += batch.ntokens
        tokens       += batch.ntokens

        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d, Loss: %f, Tokens per Sec: %f" % (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)

    # 시작은 [START]로 시작한다.
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)

    # 생성할 시퀀스의 최대 길이만큼 순회하면서
    for i in range(max_len - 1):
        print('ys.shape:', ys.shape)
        out = model.decode(
            memory, src_mask, ys,
            subsequent_mask(ys.size(1)).type_as(src.data)
        )
        print('out.shape:', out.shape)
        print('out[:, -1].shape:', out[:, -1].shape)

        # 마지막 타임스탭의 결과를 단어들로 바꾼다.
        prob = model.generator(out[:, -1])
        print('prob.shape:', prob.shape)

        # 가장 확률이 높은 단어를 선택한다.
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]

        # 예측된 단어를 추가하고 루프 처음으로 돌아가 다시 ys를 디코더로 입력한다.
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        print('\n')
    return ys

def data_gen2(V, batch, nbatchs):
    for i in range(nbatchs):
        # 샘플하나당 시퀀스 길이는 10으로 고정
        data = torch.from_numpy(np.random.randint(1, V - 1, size=(batch, 10)))  # 1 x 10
        data.requires_grad = False
        data[:, 0] = 1  # Start = 1

        src = data.clone() # Source Word - 1 x 10
        tgt = data.clone() # Target Word - 1 x 10
        # 뒤에 다섯개는 +1
        tgt[:, V // 2:] += 1
        yield Batch(src, tgt, 0)

if __name__ == "__main__":

    # Train the simple copy task.
    V = 11 # vocab size

    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model     = make_model(V, V, N=2) # N : stacked block

    #                   model_size, factor, warmup, optimizer
    model_opt = NoamOpt(model.src_embed[0].model_dim, 1, 1200,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(30):
        model.train()
        print(f'{epoch} epoch train')
        # 미니배치에 샘플 30개씩 20배치가 한 에폭
        run_epoch(data_gen2(V, 30, 20), model, SimpleLossCompute(model.generator, criterion, model_opt))

        print('eval')
        model.eval()
        # 미니배치에 샘플 30개씩 5배치가 한 에폭
        eval_loss = run_epoch(data_gen2(V, 30, 5), model, SimpleLossCompute(model.generator, criterion, None))
        print('eval_loss:', eval_loss, '\n')

    src = torch.LongTensor([[1, 1, 2, 2, 7, 8, 9, 1, 2, 6]])
    # 정답                   1, 1, 2, 2, 7, 9, 10, 2, 3, 7

    src_mask = torch.ones(1, 1, 10)

    # print(src.shape, src_mask.shape)
    print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
