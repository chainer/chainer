#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images, scale them to 256x256 and convert them to RGB, and make
two lists of space-separated CSV whose first column is full path to image and
second column is zero-origin label (this format is same as that used by Caffe's
ImageDataLayer).

"""

import argparse
import time

import chainer.iterators
import numpy as np

import chainerx as chx

from image_dataset import PreprocessedDataset
import resnet50


def get_imagenet(dataset_iter):
    x, t = zip(*next(dataset_iter))
    return chx.array(x), chx.array(t)


def compute_loss(y, t):
    # softmax cross entropy
    score = chx.log_softmax(y, axis=1)
    mask = (t[:, chx.newaxis] == chx.arange(
        1000, dtype=t.dtype)).astype(score.dtype)
    # TODO(beam2d): implement mean
    return -(score * mask).sum() * (1 / y.shape[0])


def evaluate(model, X_test, Y_test, eval_size, batch_size):
    N_test = X_test.shape[0] if eval_size is None else eval_size

    if N_test > X_test.shape[0]:
        raise ValueError(
            'Test size can be no larger than {}'.format(X_test.shape[0]))

    with chx.no_backprop_mode():
        total_loss = chx.array(0, dtype=chx.float32)
        num_correct = chx.array(0, dtype=chx.int64)
        for i in range(0, N_test, batch_size):
            x = X_test[i:min(i + batch_size, N_test)]
            t = Y_test[i:min(i + batch_size, N_test)]

            y = model(x)
            total_loss += compute_loss(y, t) * batch_size
            num_correct += (y.argmax(axis=1).astype(t.dtype)
                            == t).astype(chx.int32).sum()

    mean_loss = float(total_loss) / N_test
    accuracy = int(num_correct) / N_test
    return mean_loss, accuracy


def main():
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument(
        '--batchsize', '-B', type=int, default=32,
        help='Learning minibatch size')
    parser.add_argument(
        '--epoch', '-E', type=int, default=10,
        help='Number of epochs to train')
    parser.add_argument(
        '--iteration', '-I', type=int, default=None,
        help='Number of iterations to train. Epoch is ignored if specified.')
    parser.add_argument(
        '--loaderjob', '-j', type=int,
        help='Number of parallel data loading processes')
    parser.add_argument(
        '--mean', '-m', default='mean.npy',
        help='Mean file (computed by compute_mean.py)')
    parser.add_argument(
        '--root', '-R', default='.',
        help='Root directory path of image files')
    parser.add_argument(
        '--val_batchsize', '-b', type=int, default=250,
        help='Validation minibatch size')
    parser.set_defaults(test=False)
    parser.add_argument(
        '--device', '-d', default='native', help='Device to use')

    args = parser.parse_args()

    chx.set_default_device(args.device)
    batch_size = args.batchsize
    eval_size = args.val_batchsize

    # Prepare model
    model = resnet50.ResNet50()

    # Prepare datasets and mean file
    mean = np.load(args.mean)
    train = PreprocessedDataset(args.train, args.root, mean, model.insize)
    test = PreprocessedDataset(args.val, args.root, mean, model.insize, False)
    train_iter = chainer.iterators.MultiprocessIterator(
        train, batch_size, n_processes=args.loaderjob)
    test_iter = chainer.iterators.MultiprocessIterator(
        test, eval_size, n_processes=args.loaderjob)

    N = len(train)

    # Train
    model.require_grad()

    it = 0
    epoch = 0
    is_finished = False
    start = time.time()

    while not is_finished:

        for i in range(0, N // batch_size):
            x, t = get_imagenet(train_iter)
            y = model(x)
            loss = compute_loss(y, t)

            loss.backward()
            model.update(lr=0.01)

            it += 1
            if args.iteration is not None:
                x_test, t_test = get_imagenet(test_iter)
                mean_loss, accuracy = evaluate(
                    model, x_test, t_test, eval_size, batch_size)
                elapsed_time = time.time() - start
                print(
                    'iteration {}... loss={},\taccuracy={},\telapsed_time={}'
                    .format(it, mean_loss, accuracy, elapsed_time))
                if it >= args.iteration:
                    is_finished = True
                    break

        epoch += 1
        if args.iteration is None:
            x_test, t_test = get_imagenet(test_iter)
            mean_loss, accuracy = evaluate(
                model, x_test, t_test, eval_size, batch_size)
            elapsed_time = time.time() - start
            print(
                'epoch {}... loss={},\taccuracy={},\telapsed_time={}'
                .format(epoch, mean_loss, accuracy, elapsed_time))
            if epoch >= args.epoch:
                is_finished = True


if __name__ == '__main__':
    main()
