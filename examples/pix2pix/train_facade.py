#!/usr/bin/env python

from __future__ import print_function

import argparse
import sys
import warnings

import numpy

import chainer
from chainer import training
from chainer.training import extensions
import chainerx

from facade_dataset import FacadeDataset
from facade_visualizer import out_image

from net import Decoder
from net import Discriminator
from net import Encoder

from updater import FacadeUpdater


def main():
    parser = argparse.ArgumentParser(
        description='chainer implementation of pix2pix')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--dataset', '-i', default='./facade/base',
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', type=str,
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--snapshot_interval', type=int, default=1000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=100,
                        help='Interval of displaying log to console')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    if chainer.get_dtype() == numpy.float16:
        warnings.warn(
            'This example may cause NaN in FP16 mode.', RuntimeWarning)

    device = chainer.get_device(args.device)
    if device.xp is chainerx:
        sys.stderr.write('This example does not support ChainerX devices.\n')
        sys.exit(1)

    print('Device: {}'.format(device))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    device.use()

    # Set up a neural network to train
    enc = Encoder(in_ch=12)
    dec = Decoder(out_ch=3)
    dis = Discriminator(in_ch=12, out_ch=3)

    enc.to_device(device)
    dec.to_device(device)
    dis.to_device(device)

    # Setup an optimizer
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
        return optimizer
    opt_enc = make_optimizer(enc)
    opt_dec = make_optimizer(dec)
    opt_dis = make_optimizer(dis)

    train_d = FacadeDataset(args.dataset, data_range=(1, 300))
    test_d = FacadeDataset(args.dataset, data_range=(300, 379))
    train_iter = chainer.iterators.SerialIterator(train_d, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_d, args.batchsize)

    # Set up a trainer
    updater = FacadeUpdater(
        models=(enc, dec, dis),
        iterator={
            'main': train_iter,
            'test': test_iter},
        optimizer={
            'enc': opt_enc, 'dec': opt_dec,
            'dis': opt_dis},
        device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(extensions.snapshot(
        filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        enc, 'enc_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dec, 'dec_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        dis, 'dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'enc/loss', 'dec/loss', 'dis/loss',
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(
        out_image(
            updater, enc, dec,
            5, 5, args.seed, args.out),
        trigger=snapshot_interval)

    if args.resume is not None:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
