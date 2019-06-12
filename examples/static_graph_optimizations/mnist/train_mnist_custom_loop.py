"""MNIST example with static subgraph optimizations.

This is a version of the Chainer MNIST example that has been modified
to support the static subgraph optimizations feature. Note that
the code is mostly unchanged except for the addition of the
`@static_graph` decorator to the model chain's `__call__()` method.

This code is a custom loop version of train_mnist.py. That is, we train
models without using the Trainer class in chainer and instead write a
training loop that manually computes the loss of minibatches and
applies an optimizer to update the model.
"""
from __future__ import print_function

import argparse
import warnings

import numpy

import chainer
from chainer import configuration
from chainer.dataset import convert
import chainer.links as L
from chainer import serializers
import chainerx

import train_mnist


def run_train_loop(
        optimizer, train_iter, test_iter, train_count, test_count, epoch,
        device):
    model = optimizer.target

    sum_accuracy = 0
    sum_loss = 0
    while train_iter.epoch < epoch:
        batch = train_iter.next()
        x_array, t_array = convert.concat_examples(batch, device)
        x = chainer.Variable(x_array)
        t = chainer.Variable(t_array, requires_grad=False)
        optimizer.update(model, x, t)
        sum_loss += float(model.loss.array) * len(t)
        sum_accuracy += float(model.accuracy.array) * len(t)

        if train_iter.is_new_epoch:
            print('epoch: ', train_iter.epoch)
            print('train mean loss: {}, accuracy: {}'.format(
                sum_loss / train_count, sum_accuracy / train_count))
            # evaluation
            sum_accuracy = 0
            sum_loss = 0
            # It is good practice to turn off train mode during evaluation.
            with configuration.using_config('train', False):
                for batch in test_iter:
                    x_array, t_array = convert.concat_examples(
                        batch, device)
                    x = chainer.Variable(x_array)
                    t = chainer.Variable(t_array, requires_grad=False)
                    loss = model(x, t)
                    sum_loss += float(loss.array) * len(t)
                    sum_accuracy += float(model.accuracy.array) * len(t)

            test_iter.reset()
            print('test mean  loss: {}, accuracy: {}'.format(
                sum_loss / test_count, sum_accuracy / test_count))
            sum_accuracy = 0
            sum_loss = 0


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--model', '-m', default='MLP',
                        help='Choose the model: MLP or MLPSideEffect')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    if chainer.get_dtype() == numpy.float16:
        warnings.warn(
            'This example may cause NaN in FP16 mode.', RuntimeWarning)

    device = chainer.get_device(args.device)

    print('Device: {}'.format(device))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    device.use()

    # Set up a neural network to train
    if args.model == 'MLP':
        model = L.Classifier(train_mnist.MLP(args.unit, 10))
    elif args.model == 'MLPSideEffect':
        model = L.Classifier(train_mnist.MLPSideEffect(args.unit, 10))
    model.to_device(device)

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train_count = len(train)
    test_count = len(test)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    if device.xp is not chainerx:
        run_train_loop(
            optimizer, train_iter, test_iter, train_count, test_count,
            args.epoch, device)
    else:
        warnings.warn(
            'Static subgraph optimization does not support ChainerX and will'
            ' be disabled.', UserWarning)
        with chainer.using_config('use_static_graph', False):
            run_train_loop(
                optimizer, train_iter, test_iter, train_count, test_count,
                args.epoch, device)

    # Save the model and the optimizer
    print('save the model')
    serializers.save_npz('mlp.model', model)
    print('save the optimizer')
    serializers.save_npz('mlp.state', optimizer)


if __name__ == '__main__':
    main()
