from __future__ import print_function

import argparse
import json
import numpy as np

import chainer
from chainer.dataset.convert import concat_examples
from chainer import serializers

import nets


def main():
    parser = argparse.ArgumentParser(description='CapsNet: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=256)
    parser.add_argument('--decay', '-d', type=float, default=0.95)
    parser.add_argument('--epoch', '-e', type=int, default=500)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--seed', '-s', type=int, default=789)
    parser.add_argument('--reconstruct', '--recon', action='store_true')
    parser.add_argument('--save')
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    # Set up a neural network to train
    np.random.seed(args.seed)
    model = nets.CapsNet(use_reconstruction=args.reconstruct)
    if args.gpu >= 0:
        # Make a speciied GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU
    np.random.seed(args.seed)
    model.xp.random.seed(args.seed)

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=1e-3)
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(ndim=3)
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, 100,
                                                 repeat=False, shuffle=False)

    def report(epoch, result):
        mode = 'train' if chainer.config.train else 'test '
        print('epoch {:2d}\t{} mean loss: {}, accuracy: {}'.format(
            train_iter.epoch, mode, result['mean_loss'], result['accuracy']))
        if args.reconstruct:
            print('\t\t\tclassification: {}, reconstruction: {}'.format(
                result['cls_loss'], result['rcn_loss']))

    best = 0.
    best_epoch = 0
    print('TRAINING starts')
    while train_iter.epoch < args.epoch:
        batch = train_iter.next()
        x, t = concat_examples(batch, args.gpu)
        optimizer.update(model, x, t)

        # evaluation
        if train_iter.is_new_epoch:
            result = model.pop_results()
            report(train_iter.epoch, result)

            with chainer.no_backprop_mode():
                with chainer.using_config('train', False):
                    for batch in test_iter:
                        x, t = concat_examples(batch, args.gpu)
                        loss = model(x, t)
                    result = model.pop_results()
                    report(train_iter.epoch, result)
            if result['accuracy'] > best:
                best, best_epoch = result['accuracy'], train_iter.epoch
                serializers.save_npz(args.save, model)

            optimizer.alpha *= args.decay
            optimizer.alpha = max(optimizer.alpha, 1e-5)
            print('\t\t# optimizer alpha', optimizer.alpha)
            test_iter.reset()
    print('Finish: Best accuray: {} at {} epoch'.format(best, best_epoch))


if __name__ == '__main__':
    main()
