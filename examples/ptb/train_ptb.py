#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import print_function
import argparse
import math

import six

import chainer
from chainer import datasets
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

    def __call__(self, inputs, optimizer):
        x, t = inputs
        ret = {}

        self.loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))
        self.count += 1
        if self.count % self._sequence_len == 0:  # Run truncated BPTT
            optimizer.target.zerograds()
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
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='learning minibatch size')
    parser.add_argument('--bproplen', '-l', type=int, default=35,
                        help='length of truncated BPTT')
    parser.add_argument('--epoch', '-e', default=39, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--gradclip', '-c', type=int, default=5,
                        help='gradient norm threshold to clip')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', default=650, type=int,
                        help='number of units')
    args = parser.parse_args()

    batchsize = args.batchsize
    sequence_len = args.bproplen

    train_baseset = datasets.PTBWordsTraining()
    valset = datasets.PTBWordsValidation()
    n_vocab = valset.n_vocab
    if args.test:
        train_baseset = datasets.SubDataset(train_baseset, 0, 100)
        valset = datasets.SubDataset(valset, 0, 100)
    trainset = ParallelSequentialLoader(train_baseset, batchsize)

    model = L.Classifier(RNNLM(n_vocab, args.unit))
    model.compute_accuracy = False  # we only want the perplexity
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    trainer = chainer.Trainer(
        trainset, model, optimizers.SGD(lr=1),
        updater=TruncatedBPTTUpdater(sequence_len), batchsize=batchsize,
        epoch=args.epoch, device=args.gpu)
    trainer.optimizer.add_hook(chainer.optimizer.GradientClipping(
        args.gradclip))

    trainer.extend(extensions.Evaluator(
        valset, model, prepare=evaluation_prepare, device=args.gpu))

    trainer.extend(extensions.PrintResult(
        trigger=(250, 'iteration'), postprocess=compute_perplexity))
    trainer.extend(extensions.Snapshot())

    if args.resume:
        serializers.load_npz(args.resume, trainer)
    trainer.run(out='result')

    print('evaluating on test set...')
    testset = datasets.PTBWordsTest()
    if args.test:
        testset = datasets.SubDataset(testset, 0, 100)
    evaluator = extensions.Evaluator(
        testset, model, prepare=evaluation_prepare, device=args.gpu)
    result = evaluator(trainer)
    print('test perplexity:', math.exp(float(result['loss'])))


if __name__ == '__main__':
    main()
