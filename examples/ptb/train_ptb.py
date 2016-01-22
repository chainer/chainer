#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import print_function
import argparse
import collections
import math

import six

import chainer
from chainer.datasets import ptb_words
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer.trainer import extensions
from chainer.utils import summary


class ParallelSequentialLoader(chainer.Dataset):

    """Data adapter that loads words starting from multiple points in parallel.
    """
    def __init__(self, baseset, batchsize):
        self._baseset = baseset
        self._batchsize = batchsize
        self._gap = len(self._baseset) // batchsize

    def __len__(self):
        return len(self._baseset)

    def __getitem__(self, i):
        index = i % self._batchsize * self._gap + i // self._batchsize
        return self._baseset[index]


class RNNLM(chainer.Chain):

    """Recurrent neural net language model for Penn Tree Bank corpus.

    This is an example of deep LSTM networks for inputs of infinite lenghts.

    """
    def __init__(self, n_vocab, n_units, train=True):
        super(RNNLM, self).__init__(
            embed=L.EmbedID(n_vocab, n_units),
            l1=L.LSTM(n_units, n_units),
            l2=L.LSTM(n_units, n_units),
            l3=L.Linear(n_units, n_vocab),
        )
        self.train = train

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0, train=self.train))
        h2 = self.l2(F.dropout(h1, train=self.train))
        y = self.l3(F.dropout(h2, train=self.train))
        return y


def compute_perplexity(means):
    for name_to_mean in means.values():
        if 'loss' not in name_to_mean:
            continue
        perplexity = math.exp(name_to_mean['loss'])
        name_to_mean['perplexity'] = perplexity


def evaluation_prepare(model):
    model.predictor.train = False
    model.predictor.reset_state()


class TruncatedBPTTUpdater(chainer.trainer.Updater):

    def __init__(self, sequence_len):
        self.loss = 0
        self.count = 0
        self._sequence_len = sequence_len

    def __call__(self, inputs, model, optimizer):
        x, t = inputs
        ret = {}

        self.loss += model(chainer.Variable(x), chainer.Variable(t))
        self.count += 1
        if self.count % self._sequence_len == 0:  # Run truncated BPTT
            model.zerograds()
            self.loss.backward()
            self.loss.unchain_backward()  # truncate
            optimizer.update()

            ret['loss'] = self.loss.data / self._sequence_len
            self.loss = 0

        return ret


class ResetState(chainer.trainer.Extension):
    def __init__(self, target):
        self.target = target

    def __call__(self, **kwargs):
        self.target.reset_state()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    batchsize = 20
    sequence_len = 35

    trainset = ParallelSequentialLoader(
        ptb_words.PTBWordsTraining(), batchsize)
    valset = ptb_words.PTBWordsValidation()

    model = L.Classifier(RNNLM(valset.n_vocab, 650))
    model.compute_accuracy = False  # we only want the perplexity

    trainer = chainer.Trainer(
        trainset, model, optimizers.SGD(lr=1),
        updater=TruncatedBPTTUpdater(sequence_len), batchsize=batchsize,
        epoch=39)
    trainer.optimizer.add_hook(chainer.optimizer.GradientClipping(5))

    trainer.extend(extensions.Evaluator(
        ptb_words.PTBWordsValidation(), model, prepare=evaluation_prepare))

    trainer.extend(extensions.PrintResult(
        trigger=(250, 'iteration'), postprocess=compute_perplexity))
    trainer.extend(extensions.Snapshot())

    if args.resume:
        serializers.load_npz(args.resume, trainer)
    if args.gpu >= 0:
        trainer.to_gpu(args.gpu)
    trainer.run(out='result')


if __name__ == '__main__':
    main()
