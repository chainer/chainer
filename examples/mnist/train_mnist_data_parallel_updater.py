#!/usr/bin/env python
import argparse

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions
import chainerx
import sys

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
    parser.add_argument('--devices', '-d', type=str, nargs='*',
                        default=['0', '1', '2', '3'],
                        help='Device specifiers. Either ChainerX device '
                        'specifiers or integers. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--ljob', '-j', type=int, default=4,
                        help='Number of parallel data loading processes')
    args = parser.parse_args()

    devices = tuple([chainer.get_device(d) for d in args.devices])
    if any(device.xp is chainerx for device in devices):
        sys.stderr.write('This example does not support ChainerX devices.\n')
        sys.exit(1)

    print('Devices: {}'.format(args.devices))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    model = L.Classifier(train_mnist.MLP(args.unit, 10))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train, test = chainer.datasets.get_mnist()

    train_iters = [
        chainer.iterators.MultiprocessIterator(i,
                                               args.batchsize,
                                               n_processes=args.ljob)
        for i in chainer.datasets.split_dataset_n_random(train, args.ljob)]
    test_iter = chainer.iterators.MultiprocessIterator(
        test, args.batchsize, repeat=False, n_processes=args.ljob)

    updater = training.updaters.MultiprocessParallelUpdater(train_iters,
                                                            optimizer,
                                                            devices=(devices))

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=devices[0]))
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
