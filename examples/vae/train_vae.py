#!/usr/bin/env python
"""Chainer example: train a VAE on MNIST
"""
import argparse
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
import chainerx

import net

import matplotlib
matplotlib.use('Agg')


def main():
    parser = argparse.ArgumentParser(description='Chainer example: VAE')
    parser.add_argument('--initmodel', '-m', type=str,
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', type=str,
                        help='Resume the optimization from snapshot')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', default='results',
                        help='Directory to output the result')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--dim-z', '-z', default=20, type=int,
                        help='dimension of encoded vector')
    parser.add_argument('--dim-h', default=500, type=int,
                        help='dimension of hidden layer')
    parser.add_argument('--beta', default=1.0, type=float,
                        help='Regularization coefficient for '
                             'the second term of ELBO bound')
    parser.add_argument('--k', '-k', default=1, type=int,
                        help='Number of Monte Carlo samples used in '
                             'encoded vector')
    parser.add_argument('--binary', action='store_true',
                        help='Use binarized MNIST')
    parser.add_argument('--batch-size', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    if chainer.get_dtype() == np.float16:
        warnings.warn(
            'This example may cause NaN in FP16 mode.', RuntimeWarning)

    device = chainer.get_device(args.device)
    device.use()

    print('Device: {}'.format(device))
    print('# dim z: {}'.format(args.dim_z))
    print('# Minibatch-size: {}'.format(args.batch_size))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Prepare VAE model, defined in net.py
    encoder = net.make_encoder(784, args.dim_z, args.dim_h)
    decoder = net.make_decoder(784, args.dim_z, args.dim_h,
                               binary_check=args.binary)
    prior = net.make_prior(args.dim_z)
    avg_elbo_loss = net.AvgELBOLoss(encoder, decoder, prior,
                                    beta=args.beta, k=args.k)
    avg_elbo_loss.to_device(device)

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(avg_elbo_loss)

    # If initial parameters are given, initialize the model with them.
    if args.initmodel is not None:
        chainer.serializers.load_npz(args.initmodel, avg_elbo_loss)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(withlabel=False)

    if args.binary:
        # Binarize dataset
        train = (train >= 0.5).astype(np.float32)
        test = (test >= 0.5).astype(np.float32)

    if args.test:
        train, _ = chainer.datasets.split_dataset(train, 100)
        test, _ = chainer.datasets.split_dataset(test, 100)

    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, args.batch_size,
                                                 repeat=False, shuffle=False)

    # Set up an updater. StandardUpdater can explicitly specify a loss function
    # used in the training with 'loss_func' option
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=device, loss_func=avg_elbo_loss)

    # Set up the trainer and extensions.
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(
        test_iter, avg_elbo_loss, device=device))
    # TODO(niboshi): Temporarily disabled for chainerx. Fix it.
    if device.xp is not chainerx:
        trainer.extend(extensions.DumpGraph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/reconstr', 'main/kl_penalty', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    # If snapshot file is given, resume the training.
    if args.resume is not None:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    # Save images for demonstration
    save_images(device, encoder, decoder, train, test, prior, args.out)


# Saves 3x3 tiled image
def save3x3(x, filename):
    numpy_device = chainer.get_device('@numpy')
    fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
    for ai, xi in zip(ax.flatten(), x):
        im = xi.reshape(28, 28)
        im = numpy_device.send(im)
        ai.imshow(im)
    fig.savefig(filename)


# Saves reconstruction images using:
# - training image samples
# - test image samples
# - randomly sampled values of z
def save_images(device, encoder, decoder, train, test, prior, out_dir):

    # Training samples
    train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
    x = device.send(np.asarray(train[train_ind]))
    with chainer.using_config('train', False):
        with chainer.no_backprop_mode():
            z = encoder(x).mean
            y = decoder(z, inference=True).mean
            y = y.array
    save3x3(x, os.path.join(out_dir, 'train'))
    save3x3(y, os.path.join(out_dir, 'train_reconstructed'))

    # Test samples
    test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
    x = device.send(np.asarray(test[test_ind]))
    with chainer.using_config('train', False):
        with chainer.no_backprop_mode():
            z = encoder(x).mean
            y = decoder(z, inference=True).mean
            y = y.array
    save3x3(x, os.path.join(out_dir, 'test'))
    save3x3(y, os.path.join(out_dir, 'test_reconstructed'))

    # Draw images from 9 randomly sampled values of z
    z = prior().sample(9)
    with chainer.using_config('train', False):
        with chainer.no_backprop_mode():
            y = decoder(z, inference=True).mean
            y = y.array
    save3x3(y, os.path.join(out_dir, 'sampled'))


if __name__ == '__main__':
    main()
