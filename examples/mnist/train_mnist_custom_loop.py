#!/usr/bin/env python
from __future__ import print_function
import argparse
import copy

import chainer
from chainer import configuration
from chainer.dataset import convert
import chainer.links as L
from chainer import serializers
from chainer.utils.training import IteratorProgressBar

from models import MLP


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
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    model = L.Classifier(MLP(args.unit, 10))
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    train_count = len(train)
    test_count = len(test)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    sum_accuracy = 0
    sum_loss = 0
    # This is a timer utility that displays a progress bar using information
    # from the supplied iterator. It must be called each iteration.
    train_progress = IteratorProgressBar(train_iter,
                                         training_length=(args.epoch, 'epoch'))
    while train_iter.epoch < args.epoch:
        batch = train_iter.next()
        if train_progress():
            # You can periodically print additional progress updates here.
            # The default interval returns true once per second.
            # To obtain the same behavior without the progress bar display,
            # use a TimerUtility instead.
            pass
        x_array, t_array = convert.concat_examples(batch, args.gpu)
        x = chainer.Variable(x_array)
        t = chainer.Variable(t_array)
        optimizer.update(model, x, t)
        sum_loss += float(model.loss.data) * len(t.data)
        sum_accuracy += float(model.accuracy.data) * len(t.data)

        if train_iter.is_new_epoch:
            print('train mean loss={}, accuracy={}'.format(
                    sum_loss / train_count, sum_accuracy / train_count))
            # Evaluate the model.
            sum_accuracy = 0
            sum_loss = 0
            # It is good practice to turn off train mode during evaluation.
            with configuration.using_config('train', False):
                for batch in copy.copy(test_iter):
                    x_array, t_array = convert.concat_examples(batch, args.gpu)
                    x = chainer.Variable(x_array)
                    t = chainer.Variable(t_array)
                    loss = model(x, t)
                    sum_loss += float(loss.data) * len(t.data)
                    sum_accuracy += float(model.accuracy.data) * len(t.data)

            print('test mean loss={}, accuracy={}'.format(
                sum_loss / test_count, sum_accuracy / test_count))
            sum_accuracy = 0
            sum_loss = 0

    # Save the model and the optimizer
    print('save the model')
    serializers.save_npz('mlp.model', model)
    print('save the optimizer')
    serializers.save_npz('mlp.state', optimizer)


if __name__ == '__main__':
    main()
