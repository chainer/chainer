from __future__ import division

import unittest

import numpy as np

import chainer
import chainer.testing
from chainer.training import extension
import chainermn
from chainermn.extensions import ObservationAggregator


class DummyChain(chainer.Chain):

    def __init__(self):
        super(DummyChain, self).__init__()
        with self.init_scope():
            self.l = chainer.links.Linear(None, 1)

    def forward(self, x):
        return chainer.functions.sum(self.l(x))
#        xp = chainer.backend.get_array_module(x)
#        print(xp)
#        return chainer.Variable(x, grad=xp.array([0]))


@chainer.testing.parameterize(*chainer.testing.product({
    'use_cupy': [False, True],
    'use_chainer_variable': [False, True],
    'communicate_interval': [1, 2],
}))
class TestObservationAggregator(unittest.TestCase):

    def setUp(self):
        if self.use_cupy:
            comm_name = 'pure_nccl'
            import cupy
            self.xp = cupy
        else:
            comm_name = 'naive'
            self.xp = np
        self.communicator = chainermn.create_communicator(comm_name)

        if self.use_cupy:
            cupy.cuda.Device(self.communicator.intra_rank).use()

    def test_observation_aggregator(self):
        model = DummyChain()
        if self.use_cupy:
            model.to_gpu()
        comm = self.communicator

        optimizer = chainermn.create_multi_node_optimizer(
            chainer.optimizers.Adam(), self.communicator)
        optimizer.setup(model)

        train = self.xp.random.rand(10, 1).astype(np.float32)
        train_iter = chainer.iterators.SerialIterator(train,
                                                      batch_size=1,
                                                      repeat=True,
                                                      shuffle=True)

        updater = chainer.training.StandardUpdater(train_iter, optimizer)

        trainer = chainer.training.Trainer(updater, (1, 'epoch'))

        @extension.make_extension(
            trigger=(1, 'iteration'), priority=extension.PRIORITY_WRITER)
        def rank_reporter(trainer):
            tmp = self.xp.asarray(comm.rank, dtype=np.float32)
            if self.use_chainer_variable:
                tmp = chainer.Variable(tmp)
            trainer.observation['rank'] = tmp

        @extension.make_extension(
            trigger=(self.communicate_interval, 'iteration'),
            priority=extension.PRIORITY_READER)
        def aggregated_rank_checker(trainer):
            actual = trainer.observation['rank-aggregated']
            if self.use_chainer_variable:
                actual = actual.data
            expected = (comm.size - 1) / 2
            chainer.testing.assert_allclose(actual, expected)

        trainer.extend(rank_reporter)
        trainer.extend(ObservationAggregator(
            comm, 'rank', 'rank-aggregated',
            comm_trigger=(self.communicate_interval, 'iteration')))
        trainer.extend(aggregated_rank_checker)

        trainer.run()
