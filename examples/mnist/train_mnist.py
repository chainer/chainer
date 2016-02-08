#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net.

"""
import argparse

import chainer
from chainer import cuda
from chainer.datasets import mnist
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import serializers
from chainer.trainer import extensions


class MnistMLP(chainer.Chain):

    """An example of multi-layer perceptron for MNIST dataset.

    This is a very simple implementation of an MLP. You can modify this code to
    build your own neural net.

    """
    def __init__(self, n_in, n_units, n_out):
        super(MnistMLP, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_out),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class MnistMLPParallel(chainer.Chain):

    """An example of model-parallel MLP.

    This chain combines four small MLPs on two different devices.

    """
    def __init__(self, n_in, n_units, n_out):
        super(MnistMLPParallel, self).__init__(
            first0=MnistMLP(n_in, n_units // 2, n_units).to_gpu(0),
            first1=MnistMLP(n_in, n_units // 2, n_units).to_gpu(1),
            second0=MnistMLP(n_units, n_units // 2, n_out).to_gpu(0),
            second1=MnistMLP(n_units, n_units // 2, n_out).to_gpu(1),
        )

    def __call__(self, x):
        # assume x is on GPU 0
        x1 = F.copy(x, 1)

        z0 = self.first0(x)
        z1 = self.first1(x1)

        # sync
        h0 = z0 + F.copy(z1, 0)
        h1 = z1 + F.copy(z0, 1)

        y0 = self.second0(F.relu(h0))
        y1 = self.second1(F.relu(h1))

        # sync
        y = y0 + F.copy(y1, 0)
        return y

    def to_gpu(self, device=None):
        # ignore it
        pass


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--net', '-n', choices=('simple', 'parallel'),
                        default='simple', help='Network type')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from given snapshot')
    args = parser.parse_args()

    if args.net == 'simple':
        model = L.Classifier(MnistMLP(784, 1000, 10))
        if args.gpu >= 0:
            model.to_gpu(args.gpu)
    else:
        args.gpu = 0
        model = L.Classifier(MnistMLPParallel(784, 1000, 10))

    trainer = chainer.create_standard_trainer(
        mnist.MnistTraining(), model, optimizers.Adam(),
        batchsize=100, epoch=20, device=args.gpu)
    trainer.extend(extensions.Evaluator(
        mnist.MnistTest(), model, batchsize=100, device=args.gpu))
    trainer.extend(extensions.ComputationalGraph(model))

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
