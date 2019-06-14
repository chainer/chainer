#!/usr/bin/env python
import argparse

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions
import chainerx

import train_mnist


def main():
    # This script is almost identical to train_mnist.py. The only difference is
    # that this script uses data-parallel computation on two GPUs.
    # See train_mnist.py for more details.
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=400,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--out', '-o', default='result_data_parallel',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
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

    device0.use()

    model = L.Classifier(train_mnist.MLP(args.unit, 10))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist()
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # ParallelUpdater implements the data-parallel gradient computation on
    # multiple devices. It accepts "devices" argument that specifies which
    # device to use.
    updater = training.updaters.ParallelUpdater(
        train_iter,
        optimizer,
        # The device of the name 'main' is used as a "master", while others are
        # used as slaves. Names other than 'main' are arbitrary.
        devices={'main': device0, 'second': device1},
    )
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
