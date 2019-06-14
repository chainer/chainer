#!/usr/bin/env python
import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import chainerx

import train_mnist


# Network definition
class ParallelMLP(chainer.Chain):

    def __init__(self, n_units, n_out, device0, device1):
        super(ParallelMLP, self).__init__()
        self.device0 = device0
        self.device1 = device1

        with self.init_scope():
            # the input size, 784, is inferred
            self.first0 = train_mnist.MLP(n_units // 2, n_units)
            self.first1 = train_mnist.MLP(n_units // 2, n_units)
            self.first0.to_device(device0)
            self.first1.to_device(device1)

            # the input size, n_units, is inferred
            self.second0 = train_mnist.MLP(n_units // 2, n_out)
            self.second1 = train_mnist.MLP(n_units // 2, n_out)
            self.second0.to_device(device0)
            self.second1.to_device(device1)

    def forward(self, x):
        if self.device0 != self.device1:
            # assume x is on device0
            x1 = F.copy(x, self.device1)

            z0 = self.first0(x)
            z1 = self.first1(x1)

            # synchronize
            h0 = z0 + F.copy(z1, self.device0)
            h1 = z1 + F.copy(z0, self.device1)

            y0 = self.second0(F.relu(h0))
            y1 = self.second1(F.relu(h1))

            y = y0 + F.copy(y1, self.device0)
            return y  # output is on device0
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
    parser.add_argument('--out', '-o', default='result_model_parallel',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', default=1000, type=int,
                        help='Number of units')
    parser.add_argument('--device0', '-d', type=str, default='0',
                        help='Device specifier of the first device.'
                        'Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--device1', '-D', type=str, default='1',
                        help='Device specifier of the second device. '
                        'Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu0', '-g', dest='device0', type=int, nargs='?',
                       const=0,
                       help='First GPU ID')
    group.add_argument('--gpu1', '-G', dest='device1', type=int, nargs='?',
                       const=1,
                       help='Second GPU ID')
    args = parser.parse_args()
    device0 = chainer.get_device(args.device0)
    device1 = chainer.get_device(args.device1)

    print('Devices: {}, {}'.format(device0, device1))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # See train_mnist.py for the meaning of these lines

    model = L.Classifier(ParallelMLP(args.unit, 10, device0, device1))
    device0.use()

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist()

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, input_device=device0)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=device0))
    # TODO(niboshi): Temporarily disabled for chainerx. Fix it.
    if device0.xp is not chainerx:
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
