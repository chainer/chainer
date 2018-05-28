#!/usr/bin/env python
import argparse

import numpy

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer import graph_summary


import matplotlib
matplotlib.use('Agg')


iii = 0

# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call3_link__(self, x):
        x1 = x + 1
        x2 = x * 2
        s = F.hstack([x2, x2, x1, x2])
        xx1, xx2 = F.lstm(x, s)
        xx = F.concat([xx1, xx2])
        s = xx
        return self.l3(s)

    def __call_linka__(self, x):
        global iii
        iii += 1
        act = F.relu if iii % 2 == 0 else F.sigmoid
        h1 = act(self.l1(x))
        h2 = act(self.l2(h1))
        return self.l3(h2)

    def __call_link__(self, x):
        global iii

        g = graph_summary.current()

        g.config_node(x, data=[
            ('latest', dict(
                preprocess=lambda x: x[-1,:].reshape((28,28))[-1::-1,:],
            )),
            ('average', dict(
                data_reduce='average',
                preprocess=lambda x: x.mean(axis=0).reshape((28,28))[-1::-1,:],
                store_trigger=(1, 'epoch'),
                reset_trigger=(1, 'epoch'),
            )),
        ])
        h = self.l1(x)

        g.set_tag(h, 'hh')

        if iii % 2 == 0:
            h = F.relu(h)
            g.set_tag(h, 'h1_relu')
        else:
            h = F.sigmoid(h)
            g.set_tag(h, 'h1_sigmoid')

        with graph_summary.graph([h], 'g2') as g2:
            if iii % 2 == 0:
                h = F.relu(self.l2(h))
            else:
                h = F.sigmoid(self.l2(h))

            g2.set_output([h])

        h = self.l3(h)
        g.set_tag(h, 'output')
        g.config_node(h, data=[
            ('latest', dict(
                preprocess=lambda x: x[-1,:].reshape(1,10),
            )),
            ('average', dict(
                data_reduce='average',
                preprocess=lambda x: x[-1,:].reshape(1,10),
                store_trigger=(1, 'epoch'),
                reset_trigger=(1, 'epoch'),
            )),
        ])

        iii += 1
        return h

def _combine_ndim_mean_std(mean_std, n):
    mean, std = mean_std
    total_mean = mean.mean()
    total_var = ((std * std).sum() + ((mean - total_mean) ** 2).sum()) / std.size
    total_std = numpy.sqrt(total_var)
    return (total_mean, total_std)

def init_graph():
    data_config_set = [
        ('latest', dict(
            preprocess=lambda x: x[-1,:].reshape((25, 40)),
        )),
        ('average', dict(
            data_reduce='average',
            preprocess=lambda x: x.mean(axis=0).reshape((25, 40)),
            store_trigger=(1, 'epoch'),
            reset_trigger=(1, 'epoch'),
        )),
        ('mean-std', dict(
            data_reduce='mean-std',
            postprocess=_combine_ndim_mean_std,
            store_trigger=(100, 'iteration'),
            reset_trigger=(100, 'iteration'),
        )),
        ('percentile', dict(
            data_reduce='percentile',
            store_trigger=(100, 'iteration'),
            reset_trigger=(100, 'iteration'),
        )),
    ]


    graph = graph_summary.Graph('root_graph')
    for name in ('predictor/var', 'predictor/output', 'predictor/hh', 'predictor/h1_relu', 'predictor/h1_sigmoid'):
        graph.config_node(
            name,
            data=data_config_set)
    return graph


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='Disable PlotReport extension')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = L.Classifier(MLP(args.unit, 10))
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()
    #train = train[:10]

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    graph = init_graph()
    trainer.extend(graph_summary.GraphSummary(graph, ['main/loss']))

    # Evaluate the model with the test dataset for each epoch
    #trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    #trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    """
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))
    """

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    """
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    """

    # Print a progress bar to stdout
    #trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    print("Starting graph server")
    graph_summary.run_server(graph, async=True)
    trainer.run()

    import time
    time.sleep(1000)

if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        import pdb
        pdb.post_mortem()
