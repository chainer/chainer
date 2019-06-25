#!/usr/bin/env python

import argparse
import collections
import warnings

import chainer
from chainer.training import extensions

import numpy

import babi
import memnn


def train(train_data_path, test_data_path, args):
    device = chainer.get_device(args.device)
    device.use()

    vocab = collections.defaultdict(lambda: len(vocab))
    vocab['<unk>'] = 0

    train_data = babi.read_data(vocab, train_data_path)
    test_data = babi.read_data(vocab, test_data_path)
    print('Training data: %s: %d' % (train_data_path, len(train_data)))
    print('Test data: %s: %d' % (test_data_path, len(test_data)))

    train_data = memnn.convert_data(train_data, args.max_memory)
    test_data = memnn.convert_data(test_data, args.max_memory)

    encoder = memnn.make_encoder(args.sentence_repr)
    network = memnn.MemNN(
        args.unit, len(vocab), encoder, args.max_memory, args.hop)
    model = chainer.links.Classifier(network, label_key='answer')
    opt = chainer.optimizers.Adam()

    model.to_device(device)

    opt.setup(model)

    train_iter = chainer.iterators.SerialIterator(
        train_data, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test_data, args.batchsize, repeat=False, shuffle=False)
    updater = chainer.training.StandardUpdater(train_iter, opt, device=device)
    trainer = chainer.training.Trainer(updater, (args.epoch, 'epoch'))

    @chainer.training.make_extension()
    def fix_ignore_label(trainer):
        network.fix_ignore_label()

    trainer.extend(fix_ignore_label)
    trainer.extend(extensions.Evaluator(test_iter, model, device=device))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.run()

    if args.model:
        memnn.save_model(args.model, model, vocab)


def main():
    parser = argparse.ArgumentParser(
        description='Chainer example: End-to-end memory networks')
    parser.add_argument('TRAIN_DATA',
                        help='Path to training data in bAbI dataset '
                        '(e.g. "qa1_single-supporting-fact_train.txt")')
    parser.add_argument('TEST_DATA',
                        help='Path to test data in bAbI dataset '
                        '(e.g. "qa1_single-supporting-fact_test.txt")')
    parser.add_argument('--model', '-m', default='model',
                        help='Model directory where it stores trained model')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--unit', '-u', type=int, default=20,
                        help='Number of units')
    parser.add_argument('--hop', '-H', type=int, default=3,
                        help='Number of hops')
    parser.add_argument('--max-memory', type=int, default=50,
                        help='Maximum number of memory')
    parser.add_argument('--sentence-repr',
                        choices=['bow', 'pe'], default='bow',
                        help='Sentence representation. '
                        'Select from BoW ("bow") or position encoding ("pe")')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    if chainer.get_dtype() == numpy.float16:
        warnings.warn(
            'This example may cause NaN in FP16 mode.', RuntimeWarning)

    train(args.TRAIN_DATA, args.TEST_DATA, args)


if __name__ == '__main__':
    main()
