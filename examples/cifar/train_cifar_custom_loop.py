#!/usr/bin/env python
"""Convnet example using CIFAR10 or CIFAR100 dataset

This code is a custom loop version of train_cifar.py. That is, we train
models without using the Trainer class in chainer and instead write a
training loop that manually computes the loss of minibatches and
applies an optimizer to update the model.
"""
import argparse
import os

import chainer
from chainer import configuration
from chainer.dataset import convert
import chainer.links as L
from chainer import serializers

from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

import models.VGG


def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--dataset', default='cifar10',
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.05,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.add_argument('--resume', '-r', type=str,
                        help='Directory that has `vgg.model` and `vgg.state`')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    device = chainer.get_device(args.device)
    device.use()

    print('Device: {}'.format(device))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    if args.dataset == 'cifar10':
        print('Using CIFAR10 dataset.')
        class_labels = 10
        train, test = get_cifar10()
    elif args.dataset == 'cifar100':
        print('Using CIFAR100 dataset.')
        class_labels = 100
        train, test = get_cifar100()
    else:
        raise RuntimeError('Invalid dataset choice.')

    if args.test:
        train = train[:200]
        test = test[:200]

    train_count = len(train)
    test_count = len(test)

    model = L.Classifier(models.VGG.VGG(class_labels))
    model.to_device(device)

    optimizer = chainer.optimizers.MomentumSGD(args.learnrate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    if args.resume is not None:
        resume = args.resume
        if os.path.exists(resume):
            serializers.load_npz(os.path.join(resume, 'vgg.model'), model)
            serializers.load_npz(os.path.join(resume, 'vgg.state'), optimizer)
        else:
            raise ValueError(
                '`args.resume` ("{}") is specified,'
                ' but it does not exist.'.format(resume)
            )

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    sum_acc = 0
    sum_loss = 0

    while train_iter.epoch < args.epoch:
        batch = train_iter.next()
        # Reduce learning rate by 0.5 every 25 epochs.
        if train_iter.epoch % 25 == 0 and train_iter.is_new_epoch:
            optimizer.lr *= 0.5
            print('Reducing learning rate to: {}'.format(optimizer.lr))

        x, t = convert.concat_examples(batch, device)
        optimizer.update(model, x, t)
        sum_loss += float(model.loss.array) * len(t)
        sum_acc += float(model.accuracy.array) * len(t)

        if train_iter.is_new_epoch:
            print('epoch: {}'.format(train_iter.epoch))
            print('train mean loss: {}, accuracy: {}'.format(
                sum_loss / train_count, sum_acc / train_count))
            sum_acc = 0
            sum_loss = 0
            # Enable evaluation mode.
            with configuration.using_config('train', False):
                # This is optional but can reduce computational overhead.
                with chainer.using_config('enable_backprop', False):
                    for batch in test_iter:
                        x, t = convert.concat_examples(batch, device)
                        loss = model(x, t)
                        sum_loss += float(loss.array) * len(t)
                        sum_acc += float(model.accuracy.array) * len(t)

            test_iter.reset()
            print('test mean  loss: {}, accuracy: {}'.format(
                sum_loss / test_count, sum_acc / test_count))
            sum_acc = 0
            sum_loss = 0

    # Save the model and the optimizer
    out = args.out
    if not os.path.exists(out):
        os.makedirs(out)
    print('save the model')
    serializers.save_npz(os.path.join(out, 'vgg.model'), model)
    print('save the optimizer')
    serializers.save_npz(os.path.join(out, 'vgg.state'), optimizer)


if __name__ == '__main__':
    main()
