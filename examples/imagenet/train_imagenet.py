#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images and scale them to 256x256, and make two lists of space-
separated CSV whose first column is full path to image and second column is
zero-origin label (this format is same as that used by Caffe's ImageDataLayer).

"""
import argparse
import random

import numpy as np

import chainer
from chainer import datasets
from chainer import optimizers
from chainer import serializers
from chainer.trainer import extensions
from chainer.utils import crop

import alex
import googlenet
import googlenetbn
import nin


class PreprocessedImageListDataset(datasets.ImageListDataset):

    def __init__(self, path, meanpath, cropsize, root='.', test=False):
        super(PreprocessedImageListDataset, self).__init__(path, root=root)
        self._mean = self.compute_mean_with_cache(meanpath)
        self._cropsize = cropsize, cropsize
        self._test = test

    def __getitem__(self, i):
        image, label = super(PreprocessedImageListDataset, self).__getitem__(i)
        image -= self._mean
        if self._test:
            image = crop.crop_center(image, self._cropsize)
        else:
            image = crop.crop_random(image, self._cropsize)
        image /= 255
        if not self._test:
            image = image[:, :, ::random.choice((-1, 1))]
        return image, label


def main():
    arch_map = {
        'alex': alex.Alex,
        'googlenet': googlenet.GoogLeNet,
        'googlenetbn': googlenetbn.GoogLeNetBN,
        'nin': nin.NIN,
    }
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--arch', '-a', default='nin', choices=arch_map.keys(),
                        help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Minibatch size used in training')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Minibatch size used in validation')
    parser.add_argument('--epoch', '-e', default=10, type=int,
                        help='Number of epochs to learn')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--job', '-j', default=8, type=int,
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
    model = arch_map[args.arch]()
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    # Prepare dataset
    train = datasets.MultiprocessLoader(
        PreprocessedImageListDataset(
            args.train, args.mean, model.insize, args.root), args.job)
    val = datasets.MultiprocessLoader(
        PreprocessedImageListDataset(
            args.val, args.mean, model.insize, args.root, True), args.job)

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

    if args.resume:
        serializers.load_npz(args.resume, trainer)

    trainer.run(args.out)


if __name__ == '__main__':
    main()
