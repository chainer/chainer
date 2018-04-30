#!/usr/bin/env python
"""Sample script of recursive neural networks for sentiment analysis.

This is Socher's simple recursive model, not RTNN:
  R. Socher, C. Lin, A. Y. Ng, and C.D. Manning.
  Parsing Natural Scenes and Natural Language with Recursive Neural Networks.
  in ICML2011.

"""

import argparse
import collections

import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import reporter
from chainer.training import extensions

import data


def convert_tree(vocab, exp):
    assert isinstance(exp, list) and (len(exp) == 2 or len(exp) == 3)

    if len(exp) == 2:
        label, leaf = exp
        if leaf not in vocab:
            vocab[leaf] = len(vocab)
        return {'label': int(label), 'node': vocab[leaf]}
    elif len(exp) == 3:
        label, left, right = exp
        node = (convert_tree(vocab, left), convert_tree(vocab, right))
        return {'label': int(label), 'node': node}


class RecursiveNet(chainer.Chain):

    def __init__(self, n_vocab, n_units, n_label):
        super(RecursiveNet, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.l = L.Linear(n_units * 2, n_units)
            self.w = L.Linear(n_units, n_label)

    def __call__(self, x):
        accum_loss = 0.0
        result = collections.defaultdict(lambda: 0)
        # calculate each tree in batch ``x`` because we cannot process as batch
        for tree in x:
            loss, _ = self._traverse(tree, evaluate=result)
            accum_loss += loss

        reporter.report({'loss': accum_loss}, self)
        reporter.report({'total': result['total']}, self)
        reporter.report({'correct': result['correct']}, self)
        return accum_loss

    def leaf(self, x):
        return self.embed(x)

    def node(self, left, right):
        return F.tanh(self.l(F.concat((left, right))))

    def label(self, v):
        return self.w(v)

    def _traverse(self, node, evaluate):
        if isinstance(node['node'], int):
            # leaf node
            word = self.xp.array([node['node']], np.int32)
            loss = 0
            v = self.leaf(word)
        else:
            # internal node
            left_node, right_node = node['node']
            left_loss, left = self._traverse(left_node, evaluate=evaluate)
            right_loss, right = self._traverse(right_node, evaluate=evaluate)
            v = self.node(left, right)
            loss = left_loss + right_loss

        y = self.label(v)

        label = self.xp.array([node['label']], np.int32)
        t = chainer.Variable(label)
        loss += F.softmax_cross_entropy(y, t)

        predict = cuda.to_cpu(y.data.argmax(1))
        if predict[0] == node['label']:
            evaluate['correct'] += 1
        evaluate['total'] += 1

        return loss, v


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--epoch', '-e', default=400, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--unit', '-u', default=30, type=int,
                        help='number of units')
    parser.add_argument('--batchsize', '-b', type=int, default=25,
                        help='learning minibatch size')
    parser.add_argument('--label', '-l', type=int, default=5,
                        help='number of labels')
    parser.add_argument('--epocheval', '-p', type=int, default=5,
                        help='number of epochs per evaluation')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(test=False)
    args = parser.parse_args()

    n_epoch = args.epoch       # number of epochs
    n_units = args.unit        # number of units per layer
    batchsize = args.batchsize      # minibatch size
    n_label = args.label         # number of labels
    epoch_per_eval = args.epocheval  # number of epochs per evaluation

    if args.test:
        max_size = 10
    else:
        max_size = None

    vocab = {}
    train_data = [convert_tree(vocab, tree)
                  for tree in data.read_corpus('trees/train.txt', max_size)]
    train_iter = chainer.iterators.SerialIterator(train_data, batchsize)
    test_data = [convert_tree(vocab, tree)
                 for tree in data.read_corpus('trees/test.txt', max_size)]
    test_iter = chainer.iterators.SerialIterator(
        test_data, batchsize, repeat=False, shuffle=False)

    model = RecursiveNet(len(vocab), n_units, n_label)

    if args.gpu >= 0:
        model.to_gpu()

    # Setup optimizer
    optimizer = optimizers.AdaGrad(lr=0.1)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(0.0001))

    def _convert(batch, _):
        return batch

    # Setup updater
    updater = chainer.training.StandardUpdater(
        train_iter, optimizer, device=args.gpu, converter=_convert)

    # Setup trainer and run
    trainer = chainer.training.Trainer(updater, (n_epoch, 'epoch'))
    trainer.extend(
        extensions.Evaluator(test_iter, model, device=args.gpu,
                             converter=_convert),
        trigger=(epoch_per_eval, 'epoch'))
    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.MicroAverage(
        'main/correct', 'main/total', 'main/accuracy'))
    trainer.extend(extensions.MicroAverage(
        'validation/main/correct', 'validation/main/total',
        'validation/main/accuracy'))

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.run()


if __name__ == '__main__':
    main()
