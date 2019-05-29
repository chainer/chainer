#!/usr/bin/env python
import argparse

import chainer
from chainer.datasets import TransformDataset
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions

import datasets
from model import ImageCaptionModel

import matplotlib
matplotlib.use('Agg')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='result',
                        help='Output directory')
    parser.add_argument('--resume', '-r', type=str,
                        help='Resume the training from snapshot')
    parser.add_argument('--mscoco-root', type=str, default='data',
                        help='MSOCO dataset root directory')
    parser.add_argument('--max-iters', type=int, default=50000,
                        help='Maximum number of iterations to train')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Minibatch size')
    parser.add_argument('--dropout-ratio', type=float, default=0.5,
                        help='Language model dropout ratio')
    parser.add_argument('--val-keep-quantity', type=int, default=100,
                        help='Keep every N-th validation image')
    parser.add_argument('--val-iter', type=int, default=100,
                        help='Run validation every N-th iteration')
    parser.add_argument('--log-iter', type=int, default=1,
                        help='Log every N-th iteration')
    parser.add_argument('--snapshot-iter', type=int, default=1000,
                        help='Model snapshot every N-th iteration')
    parser.add_argument('--rnn', type=str, default='nsteplstm',
                        choices=['nsteplstm', 'lstm'],
                        help='Language model layer type')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--max-caption-length', type=int, default=30,
                        help='Maxium caption length when using LSTM layer')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    device = chainer.get_device(args.device)

    print('Device: {}'.format(device))
    print()

    # Load the MSCOCO dataset. Assumes that the dataset has been downloaded
    # already using e.g. the `download.py` script
    train, val = datasets.get_mscoco(args.mscoco_root)

    # Validation samples are used to address overfitting and see how well your
    # model generalizes to yet unseen data. However, since the number of these
    # samples in MSCOCO is quite large (~200k) and thus require time to
    # evaluate, you may choose to use only a fraction of the available samples
    val = val[::args.val_keep_quantity]

    # Number of unique words that are found in the dataset
    vocab_size = len(train.vocab)

    # Instantiate the model to be trained either with LSTM layers or with
    # NStepLSTM layers
    model = ImageCaptionModel(
        vocab_size, dropout_ratio=args.dropout_ratio, rnn=args.rnn)
    model.to_device(device)

    def transform(in_data):
        # Called for each sample and applies necessary preprocessing to the
        # image such as resizing and normalizing
        img, caption = in_data
        img = model.prepare(img)
        return img, caption

    # We need to preprocess the images since their sizes may vary (and the
    # model requires that they have the exact same fixed size)
    train = TransformDataset(train, transform)
    val = TransformDataset(val, transform)

    train_iter = iterators.MultiprocessIterator(
        train, args.batch_size, shared_mem=700000)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.batch_size, repeat=False, shuffle=False, shared_mem=700000)

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    @chainer.dataset.converter()
    def converter(batch, device):
        # The converted receives a batch of input samples any may modify it if
        # necessary. In our case, we need to align the captions depending on if
        # we are using LSTM layers of NStepLSTM layers in the model.
        if args.rnn == 'lstm':
            max_caption_length = args.max_caption_length
        elif args.rnn == 'nsteplstm':
            max_caption_length = None
        else:
            raise ValueError('Invalid RNN type.')
        return datasets.converter(
            batch, device, max_caption_length=max_caption_length)

    updater = training.updater.StandardUpdater(
        train_iter, optimizer=optimizer, device=device, converter=converter)

    trainer = training.Trainer(
        updater, out=args.out, stop_trigger=(args.max_iters, 'iteration'))
    trainer.extend(
        extensions.Evaluator(
            val_iter,
            target=model,
            converter=converter,
            device=device,
        ),
        trigger=(args.val_iter, 'iteration')
    )
    trainer.extend(
        extensions.LogReport(
            ['main/loss', 'validation/main/loss'],
            trigger=(args.log_iter, 'iteration')
        )
    )
    trainer.extend(
        extensions.PlotReport(
            ['main/loss', 'validation/main/loss'],
            trigger=(args.log_iter, 'iteration')
        )
    )
    trainer.extend(
        extensions.PrintReport(
            ['elapsed_time', 'epoch', 'iteration', 'main/loss',
             'validation/main/loss']
        ),
        trigger=(args.log_iter, 'iteration')
    )

    # Save model snapshots so that later on, we can load them and generate new
    # captions for any image. This can be done in the `predict.py` script
    trainer.extend(
        extensions.snapshot(filename='snapshot_{.updater.iteration}'),
        trigger=(args.snapshot_iter, 'iteration')
    )
    trainer.extend(
        extensions.snapshot_object(model, 'model_{.updater.iteration}'),
        trigger=(args.snapshot_iter, 'iteration')
    )
    trainer.extend(extensions.ProgressBar())

    if args.resume is not None:
        chainer.serializers.load_npz(args.resume, trainer)
    trainer.run()


if __name__ == '__main__':
    main()
