#!/usr/bin/env python
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import train_mnist


# Network definition
class ParallelMLP(chainer.Chain):

    def __init__(self, n_units, n_out, gpu0, gpu1):
        super(ParallelMLP, self).__init__()
        self.gpu0 = gpu0
        self.gpu1 = gpu1

        with self.init_scope():
            # the input size, 784, is inferred
            self.first0 = train_mnist.MLP(n_units // 2, n_units)
            self.first1 = train_mnist.MLP(n_units // 2, n_units)

            if gpu0 >= 0:
                self.first0.to_gpu(gpu0)
            if gpu1 >= 0:
                self.first1.to_gpu(gpu1)

            # the input size, n_units, is inferred
            self.second0 = train_mnist.MLP(n_units // 2, n_out)
            self.second1 = train_mnist.MLP(n_units // 2, n_out)

            if gpu0 >= 0:
                self.second0.to_gpu(gpu0)
            if gpu1 >= 0:
                self.second1.to_gpu(gpu1)

    def forward(self, x):
        if self.gpu0 != self.gpu1:
            # assume x is on gpu0
            x1 = F.copy(x, self.gpu1)

            z0 = self.first0(x)
            z1 = self.first1(x1)

            # synchronize
            h0 = z0 + F.copy(z1, self.gpu0)
            h1 = z1 + F.copy(z0, self.gpu1)

            y0 = self.second0(F.relu(h0))
            y1 = self.second1(F.relu(h1))

            y = y0 + F.copy(y1, self.gpu0)
            return y  # output is on gpu0
        else:
            z0 = self.first0(x)
            z1 = self.first1(x)
            h = z0 + z1

            y0 = self.second0(F.relu(h))
            y1 = self.second1(F.relu(h))
            y = y0 + y1

            return y


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', default=20, type=int,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu0', '-g', default=0, type=int,
                        help='First GPU ID')
    parser.add_argument('--gpu1', '-G', default=1, type=int,
                        help='Second GPU ID')
    parser.add_argument('--out', '-o', default='result_model_parallel',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', default=1000, type=int,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}, {}'.format(args.gpu0, args.gpu1))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # See train_mnist.py for the meaning of these lines

    model = L.Classifier(ParallelMLP(args.unit, 10, args.gpu0, args.gpu1))
    chainer.backends.cuda.get_device_from_id(args.gpu0).use()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu0)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu0))
    trainer.extend(extensions.DumpGraph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
