import os
import tempfile
import unittest

import numpy as np

import chainer
from chainer.dataset import convert
import chainer.functions as F
import chainer.links as L
import chainer.testing
from chainer import training

import chainermn
from chainermn.extensions.checkpoint import _CheckpointStats
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


class TestCheckpoint(unittest.TestCase):

    def setUp(self):
        self.communicator = chainermn.create_communicator('naive')

    def test_stats(self):
        stats = _CheckpointStats()

        for i in range(1024):
            stats.start()
            stats.end()

        assert isinstance(stats.report(), str)

        stats = _CheckpointStats()
        assert isinstance(stats.report(), str)

    def test_filename_converters(self):
        checkpointer = create_multi_node_checkpointer(name='hoge',
                                                      comm=self.communicator,
                                                      cp_interval=23,
                                                      gc_interval=32)
        nums = [np.random.randint(4096) for _ in range(234)]
        filenames = checkpointer._filenames(nums)
        nums2 = []
        for n, r, i in checkpointer._parse_filenames(filenames):
            assert self.communicator.rank == r
            assert 'hoge' == n
            nums2.append(i)

        assert set(nums) == set(nums2)

        filenames2 = checkpointer._filenames(nums2)

        assert set(filenames) == set(filenames2)

    def setup_mnist_trainer(self, display_log=False):
        batchsize = 100
        n_units = 100

        comm = self.communicator
        model = L.Classifier(MLP(n_units, 10))

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
            optimizer
        )

        return updater, optimizer, train_iter, test_iter, model

    @chainer.testing.attr.slow
    def test_mnist_simple(self, display_log=True):
        updater, optimizer, train_iter, _, model = self.setup_mnist_trainer()

        path = tempfile.mkdtemp(dir='/tmp', prefix=__name__ + '-tmp-')
        if display_log:
            print('temporary file:', path)
        checkpointer = create_multi_node_checkpointer(name=__name__,
                                                      comm=self.communicator,
                                                      path=path)
        checkpointer.maybe_load(updater, optimizer)

        sum_accuracy = 0
        sum_loss = 0
        stop = 5
        train_count = len(train_iter.dataset)
        while train_iter.epoch < stop:
            batch = train_iter.next()
            x_array, t_array = convert.concat_examples(batch, -1)
            x = chainer.Variable(x_array)
            t = chainer.Variable(t_array)
            optimizer.update(model, x, t)

            sum_loss += float(model.loss.data) * len(t.data)
            sum_accuracy += float(model.accuracy.data) * len(t.data)

            if train_iter.is_new_epoch:
                if display_log:
                    print(updater.iteration, train_iter.epoch,
                          sum_loss / train_count, sum_accuracy / train_count)
                sum_loss = 0
                sum_accuracy = 0

                checkpointer.save(updater, updater.iteration)

        if display_log:
            print(self.communicator.rank, checkpointer.get_stats())

        # Allocate totally different set of training tools to avoid leakage
        data_2 = self.setup_mnist_trainer()
        updater2, optimizer2, train_iter2, test_iter2, model2 = data_2
        checkpointer2 = create_multi_node_checkpointer(
            name=__name__, comm=self.communicator, path=path)
        checkpointer2.maybe_load(updater2, optimizer2)

        # Check data properly resumed
        self.assertEqual(updater.epoch, updater2.epoch)
        self.assertEqual(updater.iteration, updater2.iteration)
        # TODO(kuenishi): find a simple way to assure model equality
        # in terms of float matrix
        # self.assertEqual(model, model2)

        # Restart training
        while train_iter2.epoch < stop * 2:
            batch = train_iter2.next()
            x_array, t_array = convert.concat_examples(batch, -1)
            x = chainer.Variable(x_array)
            t = chainer.Variable(t_array)
            optimizer2.update(model2, x, t)

            sum_loss += float(model2.loss.data) * len(t.data)
            sum_accuracy += float(model2.accuracy.data) * len(t.data)

            if train_iter2.is_new_epoch:
                print(updater2.iteration, train_iter2.epoch,
                      sum_loss / train_count, sum_accuracy / train_count)
                sum_loss = 0
                sum_accuracy = 0

                checkpointer2.save(updater2, updater2.iteration)

        if display_log:
            print(self.communicator.rank, checkpointer2.get_stats())
        checkpointer2.finalize()
        checkpointer.finalize()

        # Validate training
        sum_accuracy = 0
        sum_loss = 0
        test_count = len(test_iter2.dataset)
        for batch in test_iter2:
            x_array, t_array = convert.concat_examples(batch, -1)
            x = chainer.Variable(x_array)
            t = chainer.Variable(t_array)
            loss = model2(x, t)
            sum_loss += float(loss.data) * len(t.data)
            sum_accuracy += float(model2.accuracy.data) * len(t.data)

        if display_log:
            print('test mean  loss: {}, accuracy: {}'.format(
                sum_loss / test_count, sum_accuracy / test_count))

        self.assertGreaterEqual(sum_accuracy / test_count, 0.95)
        os.removedirs(path)
