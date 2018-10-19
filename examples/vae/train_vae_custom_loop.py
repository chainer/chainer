#!/usr/bin/env python
"""Chainer example: train a VAE on MNIST
"""
import argparse
import os

import chainer
from chainer.dataset import convert
import numpy as np

import net


def main():
    parser = argparse.ArgumentParser(description='Chainer example: VAE')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--dimz', '-z', default=20, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# dim z: {}'.format(args.dimz))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Prepare VAE model, defined in net.py
    model = net.VAE(784, args.dimz, 500)
    if args.gpu >= 0:
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Initialize / Resume
    if args.initmodel:
        chainer.serializers.load_npz(args.initmodel, model)

    if args.resume:
        chainer.serializers.load_npz(args.resume, optimizer)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(withlabel=False)
    if args.test:
        train, _ = chainer.datasets.split_dataset(train, 100)
        test, _ = chainer.datasets.split_dataset(test, 100)
    train_count = len(train)
    test_count = len(test)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    while train_iter.epoch < args.epoch:
        sum_loss = 0
        sum_rec_loss = 0

        batch = train_iter.next()
        x_array = convert.concat_examples(batch, args.gpu)
        x = chainer.Variable(x_array)
        # Update model based on the loss function
        # defined by model.get_loss_func()
        optimizer.update(model.get_loss_func(), x)

        sum_loss += float(model.loss.array) * len(x)
        sum_rec_loss += float(model.rec_loss.array) * len(x)

        if train_iter.is_new_epoch:
            print('train mean loss={}, mean reconstruction loss={}'
                  .format(sum_loss / train_count, sum_rec_loss / train_count))

            # evaluation
            sum_loss = 0
            sum_rec_loss = 0
            for batch in test_iter:
                x_array = convert.concat_examples(batch, args.gpu)
                x = chainer.Variable(x_array)
                loss_func = model.get_loss_func(k=10)
                loss_func(x)
                sum_loss += float(model.loss.array) * len(x)
                sum_rec_loss += float(model.rec_loss.array) * len(x)

            test_iter.reset()
            print('test mean loss={}, mean reconstruction loss={}'
                  .format(sum_loss / test_count, sum_rec_loss / test_count))

    # Note that os.makedirs(path, exist_ok=True) can be used
    # if this script only supports python3
    if not os.path.exists(args.out):
        os.mkdir(args.out)

    # Save the model and the optimizer
    print('save the model')
    chainer.serializers.save_npz(
        os.path.join(args.out, 'mlp.model'), model)
    print('save the optimizer')
    chainer.serializers.save_npz(
        os.path.join(args.out, 'mlp.state'), optimizer)

    # Visualize the results
    def save_images(x, filename):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
        for ai, xi in zip(ax.flatten(), x):
            ai.imshow(xi.reshape(28, 28))
        fig.savefig(filename)

    model.to_cpu()
    train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
    x = chainer.Variable(np.asarray(train[train_ind]))
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x1 = model(x)
    save_images(x.array, os.path.join(args.out, 'train'))
    save_images(x1.array, os.path.join(args.out, 'train_reconstructed'))

    test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
    x = chainer.Variable(np.asarray(test[test_ind]))
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x1 = model(x)
    save_images(x.array, os.path.join(args.out, 'test'))
    save_images(x1.array, os.path.join(args.out, 'test_reconstructed'))

    # draw images from randomly sampled z
    z = chainer.Variable(
        np.random.normal(0, 1, (9, args.dimz)).astype(np.float32))
    x = model.decode(z)
    save_images(x.array, os.path.join(args.out, 'sampled'))


if __name__ == '__main__':
    main()
