#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import print_function
import argparse
import math
import sys
import time

import numpy as np
import six

import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np

n_epoch = 39   # number of epochs
n_units = 650  # number of units per layer
batchsize = 20   # minibatch size
bprop_len = 35   # length of truncated BPTT
grad_clip = 5    # gradient norm threshold to clip

# Prepare dataset (preliminary download dataset by ./download.py)
vocab = {}


def load_data(filename):
    global vocab, n_vocab
    words = open(filename).read().replace('\n', '<eos>').strip().split()
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    return dataset

train_data = load_data('ptb.train.txt')
valid_data = load_data('ptb.valid.txt')
test_data = load_data('ptb.test.txt')
print('#vocab =', len(vocab))


class RNNLM(chainer.DictLink):

    def __init__(self, n_vocab, n_units):
        super(RNNLM, self).__init__(
            embed=F.EmbedID(n_vocab, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=F.Linear(n_units, n_vocab),
        )

        for _, param in self.visitparams():
            param.data[:] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def forward_one_step(self, x, t, train=True):
        # Neural net architecture
        h0 = self['embed'](x)
        h1 = self['l1'](F.dropout(h0, train=train))
        h2 = self['l2'](F.dropout(h1, train=train))
        y = self['l3'](F.dropout(h2, train=train))
        return F.softmax_cross_entropy(y, t)

    def reset_state(self):
        self['l1'].reset_state()
        self['l2'].reset_state()


model = RNNLM(len(vocab), n_units)

if args.gpu >= 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()


# Setup optimizer
optimizer = optimizers.SGD(lr=1.)
optimizer.setup(model)


# Evaluation routine
def evaluate(model, dataset):
    m = model.copy()
    m.volatile = True
    sum_log_perp = xp.zeros(())
    m.reset_state()
    for i in six.moves.range(dataset.size - 1):
        x_batch = xp.asarray(dataset[i:i + 1])
        y_batch = xp.asarray(dataset[i + 1:i + 2])
        x = chainer.Variable(x_batch, volatile=True)
        t = chainer.Variable(y_batch, volatile=True)
        loss = m.forward_one_step(x, t, train=False)
        sum_log_perp += loss.data.reshape(())

    return math.exp(cuda.to_cpu(sum_log_perp) / (dataset.size - 1))


# Learning loop
whole_len = train_data.shape[0]
jump = whole_len // batchsize
cur_log_perp = xp.zeros(())
epoch = 0
start_at = time.time()
cur_at = start_at
model.reset_state()
accum_loss = chainer.Variable(xp.zeros((), dtype=np.float32))
print('going to train {} iterations'.format(jump * n_epoch))
for i in six.moves.range(jump * n_epoch):
    x_batch = xp.array([train_data[(jump * j + i) % whole_len]
                        for j in six.moves.range(batchsize)])
    y_batch = xp.array([train_data[(jump * j + i + 1) % whole_len]
                        for j in six.moves.range(batchsize)])
    x = chainer.Variable(x_batch)
    t = chainer.Variable(y_batch)
    loss_i = model.forward_one_step(x, t)
    accum_loss += loss_i
    cur_log_perp += loss_i.data.reshape(())

    if (i + 1) % bprop_len == 0:  # Run truncated BPTT
        optimizer.zero_grads()
        accum_loss.backward()
        accum_loss.unchain_backward()  # truncate
        accum_loss = chainer.Variable(xp.zeros((), dtype=np.float32))

        optimizer.clip_grads(grad_clip)
        optimizer.update()

    if (i + 1) % 10000 == 0:
        now = time.time()
        throuput = 10000. / (now - cur_at)
        perp = math.exp(cuda.to_cpu(cur_log_perp) / 10000)
        print('iter {} training perplexity: {:.2f} ({:.2f} iters/sec)'.format(
            i + 1, perp, throuput))
        cur_at = now
        cur_log_perp.fill(0)

    if (i + 1) % jump == 0:
        epoch += 1
        print('evaluate')
        now = time.time()
        perp = evaluate(model, valid_data)
        print('epoch {} validation perplexity: {:.2f}'.format(epoch, perp))
        cur_at += time.time() - now  # skip time of evaluation

        if epoch >= 6:
            optimizer.lr /= 1.2
            print('learning rate =', optimizer.lr)

    sys.stdout.flush()

# Evaluate on test dataset
print('test')
test_perp = evaluate(test_data)
print('test perplexity:', test_perp)
