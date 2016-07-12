#!/usr/bin/env python
from __future__ import print_function
import argparse
import numpy

import chainer
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import train_mnist


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('model', help='Load the model from snapshot')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('')

    # Set up a neural network to load trained model into.
    model = train_mnist.MLP(784, args.unit, 10)

    # Load the model from a snapshot
    chainer.serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()  # Copy the model to the GPU

    # Load the MNIST dataset
    train, _ = chainer.datasets.get_mnist(withlabel=False)
    dataset_iter = chainer.iterators.SerialIterator(
        train,
        args.batchsize,
        repeat=False,
        shuffle=False
    )

    # Evaluate the model with the test dataset for each epoch
    extractor = extensions.Extractor(dataset_iter, model, device=args.gpu)

    # Run the extraction
    features = extractor()
    numpy.save(
        'mnist_features.npy',
        chainer.cuda.to_cpu(features.data)
    )


if __name__ == '__main__':
    main()
