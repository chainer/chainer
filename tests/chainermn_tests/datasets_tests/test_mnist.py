# coding: utf-8

import os
import sys
import tempfile

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.testing
from chainer import training
from chainer.training import extensions

import chainermn
from chainermn.extensions.checkpoint import create_multi_node_checkpointer


class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(784, n_units)
            self.l2 = L.Linear(n_units, n_units)
            self.l3 = L.Linear(n_units, n_out)

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def check_mnist(gpu, display_log=True):
    epoch = 5
    batchsize = 100
    n_units = 100

    comm = chainermn.create_communicator('naive')
    if gpu:
        device = comm.intra_rank
        chainer.cuda.get_device_from_id(device).use()
    else:
        device = -1

    model = L.Classifier(MLP(n_units, 10))
    if gpu:
        model.to_gpu()

    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.Adam(), comm)
    optimizer.setup(model)

    if comm.rank == 0:
        train, test = chainer.datasets.get_mnist()
    else:
        train, test = None, None

    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    test = chainermn.scatter_dataset(test, comm, shuffle=True)

    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False,
                                                 shuffle=False)

    updater = training.StandardUpdater(
        train_iter,
        optimizer,
        device=device
    )

    trainer = training.Trainer(updater, (epoch, 'epoch'))

    # Wrap standard Chainer evaluators by MultiNodeEvaluator.
    evaluator = extensions.Evaluator(test_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator)

    # Add checkpointer. This is just to check checkpointing runs
    # without errors
    path = tempfile.mkdtemp(dir='/tmp', prefix=__name__ + '-tmp-')
    checkpointer = create_multi_node_checkpointer(name=__name__, comm=comm,
                                                  path=path)
    trainer.extend(checkpointer, trigger=(1, 'epoch'))

    # Some display and output extensions are necessary only for one worker.
    # (Otherwise, there would just be repeated outputs.)
    if comm.rank == 0 and display_log:
        trainer.extend(extensions.LogReport(trigger=(1, 'epoch')),
                       trigger=(1, 'epoch'))
        trainer.extend(extensions.PrintReport(['epoch',
                                               'main/loss',
                                               'validation/main/loss',
                                               'main/accuracy',
                                               'validation/main/accuracy',
                                               'elapsed_time'],
                                              out=sys.stderr),
                       trigger=(1, 'epoch'))
    trainer.run()

    err = evaluator()['validation/main/accuracy']
    assert err > 0.95

    # Check checkpointer successfully finalized snapshot directory
    assert [] == os.listdir(path)
    os.removedirs(path)


@chainer.testing.attr.slow
def test_mnist():
    check_mnist(False)


@chainer.testing.attr.gpu
def test_mnist_gpu():
    check_mnist(True)


if __name__ == '__main__':
    test_mnist()
