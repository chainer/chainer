#!/usr/bin/env python
import argparse
import re

import numpy

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import chainermn
import chainerx


# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def get_device(args, comm):
    device = '{}:{}'.format(args.device, comm.intra_rank) 
    return chainer.get_device(args.device)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--device', '-d', type=str, default='native',
                        choices=['native', 'cuda'],
                        help='Use CUDA as device')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    parser.add_argument('--communicator', type=str,
                        default='naive', help='Type of communicator')
    args = parser.parse_args()

    comm = chainermn.create_communicator(args.communicator)
    device = get_device(args, comm)

    if comm.rank == 0:
        print('Device: {}'.format(device))
        print('# unit: {}'.format(args.unit))
        print('# Minibatch-size: {}'.format(args.batchsize))
        print('# epoch: {}'.format(args.epoch))
        print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(MLP(args.unit, 10))
    model.to_device(device)
    device.use()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)

    # Load the MNIST dataset
    if comm.rank == 0:
        train, test = chainer.datasets.get_mnist()
    else:
        train, test = None, None
    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    test = chainermn.scatter_dataset(test, comm, shuffle=True)
    
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    evaluator = extensions.Evaluator(test_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator)

    if comm.rank == 0:
        # Dump a computational graph from 'loss' variable at the first iteration
        # The "main" refers to the target link of the "main" optimizer.
        # TODO(niboshi): Temporarily disabled for chainerx. Fix it.
        if device.xp is not chainerx:
            trainer.extend(extensions.DumpGraph('main/loss'))

        # Take a snapshot for each specified epoch
        frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
        trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

        # Write a log of evaluation statistics for each epoch
        trainer.extend(extensions.LogReport())

        # Save two plot images to the result dir
        if args.plot and extensions.PlotReport.available():
            trainer.extend(
                extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                      'epoch', file_name='loss.png'))
            trainer.extend(
                extensions.PlotReport(
                    ['main/accuracy', 'validation/main/accuracy'],
                    'epoch', file_name='accuracy.png'))

        # Print selected entries of the log to stdout
        # Here "main" refers to the target link of the "main" optimizer again, and
        # "validation" refers to the default name of the Evaluator extension.
        # Entries other than 'epoch' are reported by the Classifier link, called by
        # either the updater or the evaluator.
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

        # Print a progress bar to stdout
        trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
