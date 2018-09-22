#!/usr/bin/env python
"""Fully-connected neural network example using MNIST dataset

This code is a custom loop version of train_mnist.py. That is, we train
models without using the Trainer class in chainer and instead write a
training loop that manually computes the loss of minibatches and
applies an optimizer to update the model.
"""

import argparse

import chainer
from chainer import configuration
from chainer.dataset import convert
from chainer.iterators import MultiprocessIterator
import chainer.links as L
from chainer import serializers

import train_mnist


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot using model '
                             'and state files in the specified directory')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    model = L.Classifier(train_mnist.MLP(args.unit, 10))
    if args.gpu >= 0:
        # Make a speciied GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    if args.resume:
        # Resume from a snapshot
        serializers.load_npz('{}/mlp.model'.format(args.resume), model)
        serializers.load_npz('{}/mlp.state'.format(args.resume), optimizer)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train_count = len(train)
    test_count = len(test)

    with MultiprocessIterator(train, args.batchsize) as train_iter, \
        MultiprocessIterator(test, args.batchsize,
                             repeat=False, shuffle=False) as test_iter:

        sum_accuracy = 0
        sum_loss = 0

        while train_iter.epoch < args.epoch:
            batch = train_iter.next()
            x, t = convert.concat_examples(batch, args.gpu)
            optimizer.update(model, x, t)
            sum_loss += float(model.loss.array) * len(t)
            sum_accuracy += float(model.accuracy.array) * len(t)

            if train_iter.is_new_epoch:
                print('epoch: {}'.format(train_iter.epoch))
                print('train mean loss: {}, accuracy: {}'.format(
                    sum_loss / train_count, sum_accuracy / train_count))
                # evaluation
                sum_accuracy = 0
                sum_loss = 0
                # Enable evaluation mode.
                with configuration.using_config('train', False):
                    # This is optional but can reduce computational overhead.
                    with chainer.using_config('enable_backprop', False):
                        for batch in test_iter:
                            x, t = convert.concat_examples(batch, args.gpu)
                            loss = model(x, t)
                            sum_loss += float(loss.array) * len(t)
                            sum_accuracy += float(model.accuracy.array) * len(t)

                test_iter.reset()
                print('test mean  loss: {}, accuracy: {}'.format(
                    sum_loss / test_count, sum_accuracy / test_count))
                sum_accuracy = 0
                sum_loss = 0

        # Save the model and the optimizer
        print('save the model')
        serializers.save_npz('{}/mlp.model'.format(args.out), model)
        print('save the optimizer')
        serializers.save_npz('{}/mlp.state'.format(args.out), optimizer)


if __name__ == '__main__':
    main()
