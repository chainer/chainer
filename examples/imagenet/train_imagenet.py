#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images and scale them to 256x256, and make two lists of space-
separated CSV whose first column is full path to image and second column is
zero-origin label (this format is same as that used by Caffe's ImageDataLayer).

"""
import argparse

import numpy as np

import chainer
from chainer.datasets import image_dataset
from chainer.datasets import multiprocess_loader
from chainer import optimizers
from chainer.trainer import extensions
import chainer.links as L


def main():
    arch_choices = 'nin', 'alex', 'googlenet', 'googlenetbn'
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--arch', '-a', default='nin', choices=arch_choices,
                        help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Minibatch size used in training')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Minibatch size used in validation')
    parser.add_argument('--epoch', '-e', default=10, type=int,
                        help='Number of epochs to learn')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--job', '-j', default=32, type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Path to the mean file')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to save output files')
    parser.add_argument('--resume', '-R',
                        help='Resume from trainer state')
    parser.add_argument('--root', '-r', default='.',
                        help='Root directory path of image files')
    args = parser.parse_args()

    # Prepare model
    if args.arch == 'nin':
        import nin
        model = nin.NIN()
    elif args.arch == 'alex':
        import alex
        model = alex.Alex()
    elif args.arch == 'googlenet':
        import googlenet
        model = googlenet.GoogLeNet()
    elif args.arch == 'googlenetbn':
        import googlenetbn
        model = googlenetbn.GoogLeNetBN()
    else:
        raise ValueError('invalid architecture name')
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    # Prepare dataset
    train_images = image_dataset.ImageListDataset(args.train)
    try:
        with open(args.mean, 'rb') as meanfile:
            mean = np.load(meanfile)
    except IOError:
        mean = train_images.compute_mean()
        with open(args.mean, 'wb') as meanfile:
            np.write(meanfile, mean)
    train_images.add_preprocessor(image_dataset.subtract_mean(mean))
    train_images.add_preprocessor(image_dataset.crop_random(model.insize))
    train_images.add_preprocessor(image_dataset.scale(1. / 255))
    train_images.add_preprocessor(image_dataset.random_flip)

    val_images = image_dataset.ImageListDataset(args.val)
    val_images.add_preprocessor(image_dataset.subtract_mean(mean))
    val_images.add_preprocessor(image_dataset.crop_center(model.insize))
    val_images.add_preprocessor(image_dataset.scale(1. / 255))

    train = multiprocess_loader.MultiprocessLoader(train_images, args.job)
    val = multiprocess_loader.MultiprocessLoader(val_images, args.job)

    # Prepare trainer
    trainer = chainer.Trainer(
        train, model, optimizers.MomentumSGD(lr=0.01, momentum=0.9),
        batchsize=args.batchsize, epoch=args.epoch, device=args.gpu)
    trainer.optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))
    trainer.extend(extensions.LearningRateDecay(0.97))

    trainer.extend(extensions.ComputationalGraph(model))
    def set_test_mode(target):
        target.train = False
    trainer.extend(extensions.Evaluator(
        val, model, batchsize=args.val_batchsize, device=args.gpu,
        prepare=set_test_mode), trigger=(100000, 'iteration'))
    trainer.extend(extensions.PrintResult(trigger=(1000, 'iteration')))
    trainer.extend(extensions.Snapshot(), trigger=(1000, 'iteration'))

    trainer.run(args.out)


if __name__ == '__main__':
    main()
