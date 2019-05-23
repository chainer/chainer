#!/usr/bin/env python
"""Example code of learning a large scale convnet from LSVRC2012 dataset
with multiple GPUs using data parallelism.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images, scale them to 256x256 and convert them to RGB, and make
two lists of space-separated CSV whose first column is full path to image and
second column is zero-origin label (this format is same as that used by Caffe's
ImageDataLayer).

You need to install chainer with NCCL to run this example.
Please see https://github.com/nvidia/nccl#build--run .

"""
import argparse
import sys

import numpy as np

import chainer
from chainer import training
from chainer.training import extensions
from chainer.training import updaters
import chainerx

import alex
import googlenet
import googlenetbn
import nin
import resnet50
import resnext50
import train_imagenet


def main():
    archs = {
        'alex': alex.Alex,
        'googlenet': googlenet.GoogLeNet,
        'googlenetbn': googlenetbn.GoogLeNetBN,
        'nin': nin.NIN,
        'resnet50': resnet50.ResNet50,
        'resnext50': resnext50.ResNeXt50,
    }

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--arch', '-a', choices=archs.keys(),
                        default='nin', help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--devices', '-d', type=str, nargs='*',
                        default=['0', '1', '2', '3'],
                        help='Device specifiers. Either ChainerX device '
                        'specifiers or integers. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpus', '-g', dest='devices',
                       type=int, nargs='?', const=0,
                       help='GPU IDs (negative value indicates CPU)')
    args = parser.parse_args()

    devices = tuple([chainer.get_device(d) for d in args.devices])
    if any(device.xp is chainerx for device in devices):
        sys.stderr.write('This example does not support ChainerX devices.\n')
        sys.exit(1)

    # Initialize the model to train
    model = archs[args.arch]()
    if args.initmodel:
        print('Load model from {}'.format(args.initmodel))
        chainer.serializers.load_npz(args.initmodel, model)

    # Load the datasets and mean file
    mean = np.load(args.mean)
    train = train_imagenet.PreprocessedDataset(
        args.train, args.root, mean, model.insize)
    val = train_imagenet.PreprocessedDataset(
        args.val, args.root, mean, model.insize, False)
    # These iterators load the images with subprocesses running in parallel to
    # the training/validation.
    train_iters = [
        chainer.iterators.MultiprocessIterator(i,
                                               args.batchsize,
                                               n_processes=args.loaderjob)
        for i in chainer.datasets.split_dataset_n_random(train, len(devices))]
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Set up an optimizer
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)

    # Set up a trainer
    updater = updaters.MultiprocessParallelUpdater(train_iters, optimizer,
                                                   devices=devices)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    if args.test:
        val_interval = 5, 'epoch'
        log_interval = 1, 'epoch'
    else:
        val_interval = 100000, 'iteration'
        log_interval = 1000, 'iteration'

    trainer.extend(extensions.Evaluator(val_iter, model, device=devices[0]),
                   trigger=val_interval)
    trainer.extend(extensions.DumpGraph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=val_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=val_interval)
    # Be careful to pass the interval directly to LogReport
    # (it determines when to emit log rather than when to read observations)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/accuracy', 'validation/main/accuracy', 'lr'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=2))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
