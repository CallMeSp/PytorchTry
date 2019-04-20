import copy
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchtext import data, datasets

seaborn.set_context(context="talk")
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EncoderDecoder(nn.Module):
    """
    一个标准的EncoderDecoder结构。base版
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    定义标准的线性+softmax生成step
    """

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    """
    生成n个相同的层
    :param module:层
    :param N:需要生成的个数
    :return:nn.moduleList
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # pass the input (and mask) through each layer in turn
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """
    Construct a layernorm module
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        # std为标准差
        std = x.std(-1, keepdim=True)
        # 0均值标准化
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # Apply residual connection to any sublayer with the same size.
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward (defined below)
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # follow Figure 1(left) for connections
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """
    生成ｎ层带mask的decoder
    """

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    # 屏蔽掉当前位置后面的序列
    attn_shape = (1, size, size)
    # triu 返回上三角矩阵，k=1为主对角线
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# Attention(Q,K,V)=softmax(QK^T/sqrt(d_k)) * V
def attention(query, key, value, mask=None, dropout=None):
    # Compute 'Scaled Dot Product Attention'
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # 把mask==0的索引位置替换为value
        scores = scores.masked_fill(mask == 0, value=-1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        # print('in attention before dropout',p_attn)
        dropout = nn.Dropout(dropout)
        p_attn = dropout(p_attn)
        # print('in attention after dropout', p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h:并行的head数量
        :param d_model:原始word的embeddingsize或者一个encoder的输出维度
        :param dropout:dropout比例
        """
        super(MultiHeadedAttention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.dropout = dropout
        assert d_model % h == 0
        self.d_k = d_model // h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        # 有些tensor并不是占用一整块内存，而是由不同的数据块组成，
        # 而tensor的view()操作依赖于内存是整块的，这时只需要执行contiguous()这个函数，
        # 把tensor变成在内存中连续分布的形式。
        x = x.transpose(1, 2).contiguous().view(nbatches, -1,
                                                self.h * self.d_k)
        return self.linears[-1](x)


# FFN(x)=max(0,xW1+b1)W2+b2
class PositionwiseFeedForward(nn.Module):
    # Implements FFN equation.
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # 如果要使用预训练的则
        # nn.Embedding(vocab, d_model).weight.data.copy_(torch.from_numpy(W))
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        newx = x.long()
        embeddingMat = self.lut(newx) * math.sqrt(self.d_model)
        # print('x_embedding:', x.shape, embeddingMat.shape)
        return embeddingMat


class PositionalEncoding(nn.Module):
    # Implement the PE function.

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            -(torch.arange(0., d_model, 2) / d_model) * math.log(10000.0))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(-2)], requires_grad=False)
        return self.dropout(x)


def make_model(src_vocab,
               tgt_vocab,
               N=6,
               d_model=512,
               d_ff=2048,
               h=8,
               dropout=0.1):
    # construct a model from hyperparameters
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        # batch.src.shape:
        out = model.forward(batch.src, batch.trg, batch.src_mask,
                            batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        # print(i)
        # print(i, loss/batch.ntokens, float(tokens)/(time.time() - start))
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, float(tokens) / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


# 因为这是应用于翻译任务，所以这里分配batch是将长度近似的句子pair分配在一起。
def batch_size_fn(new, count, sofar):
    # keep augmenting batch and calculate total number of tokens + padding
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


class NoamOpt:
    """
    optim wrapper that implements rate
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        # update parameters and rate
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size **
                              (-0.5) * min(step **
                                           (-0.5), step * self.warmup ** (-1.5)))


class LabelSmoothing(nn.Module):
    # implement label smoothing
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # print('size:', self.size)
        # print('true_dist:', x.data)
        # fill_:将该tensor用指定的数值填充
        true_dist.fill_(self.smoothing / (self.size - 2))
        # print('true_dist_fill:', true_dist, self.smoothing / (self.size - 2))
        # 将src中的所有值按照index确定的索引写入本tensor中
        true_dist.scatter_(1, target.data.unsqueeze(1).long(), self.confidence)
        # print('true_dist_scatter:', true_dist, 'targetdataunsqueeze:', target.data.unsqueeze(1))
        true_dist[:, self.padding_idx] = 0
        # print('true_dist_0:', true_dist)
        mask = torch.nonzero(target.data == self.padding_idx)
        # print('tgtdata:', target.data, self.padding_idx, target.data == self.padding_idx, mask.data)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        # print(self.criterion(x, Variable(true_dist, requires_grad=False)))
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class Batch:
    """
    object for holding a batch of data with mask during training
    """

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_stf_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_stf_mask(tgt, pad):
        # create a mask to hide padding and future words.
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    maxlength = 10
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 300)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False).to(device)
        tgt = Variable(data, requires_grad=False).to(device)
        yield Batch(src, tgt, 0)


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss * norm


class MultiGPULossCompute:
    "A multi-gpu loss compute and train function."

    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[
                Variable(
                    o[:, i:i + chunk_size].data,
                    requires_grad=self.opt is not None)
            ] for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, target_device=self.devices[0])
            l = l.sum()[0] / normalize
            total += l.data[0]

            # Backprop loss to output of transformer
            if self.opt is not None:
                l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return total * normalize


if __name__ == '__main__':
    # Train the simple copy task
    startTime = time.time()
    V = 500
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2).to(device)
    model_opt = NoamOpt(
        model.src_embed[0].d_model, 1, 400,
        torch.optim.Adam(
            model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(5):
        print('new epoch start')
        model.train()
        tmp = data_gen(V, 64, 100)
        run_epoch(tmp, model,
                  SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(
            run_epoch(
                data_gen(V, 30, 5), model,
                SimpleLossCompute(model.generator, criterion, None)))
    endTime = time.time()
    # normal  3.9171374003092447
    print('time cost:', (endTime - startTime) / 60)
