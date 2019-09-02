from __future__ import division

import pytest

import numpy as np

import chainer
import chainer.testing
from chainer.training import extension
from chainer.backend import cuda
import chainermn
from chainermn.extensions import ObservationAggregator


class DummyChain(chainer.Chain):

    def __init__(self):
        super(DummyChain, self).__init__()
        with self.init_scope():
            self.l = chainer.links.Linear(None, 1)

    def forward(self, x):
        return chainer.functions.sum(self.l(x))


@pytest.mark.parametrize('use_chainer_variable', [False, True])
@pytest.mark.parametrize('communicate_interval', [1, 2])
def test_observation_aggregator_cpu(use_chainer_variable,
                                    communicate_interval):
    communicator = chainermn.create_communicator('naive')
    xp = np
    run_test_observation_aggregator(communicator, xp,
                                    use_chainer_variable,
                                    communicate_interval,
                                    use_cupy=False)


@pytest.mark.parametrize('use_chainer_variable', [False, True])
@pytest.mark.parametrize('communicate_interval', [1, 2])
@chainer.testing.attr.gpu
def test_observation_aggregator_gpu(use_chainer_variable,
                                    communicate_interval):
    communicator = chainermn.create_communicator('pure_nccl')
    xp = cuda.cupy
    cuda.Device(communicator.intra_rank).use()
    run_test_observation_aggregator(communicator, xp,
                                    use_chainer_variable,
                                    communicate_interval,
                                    use_cupy=True)


def run_test_observation_aggregator(comm, xp,
                                    use_chainer_variable,
                                    communicate_interval,
                                    use_cupy):
    model = DummyChain()
    if use_cupy:
        model.to_gpu()
    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.Adam(), comm)
    optimizer.setup(model)

    train = xp.random.rand(10, 1).astype(np.float32)
    train_iter = chainer.iterators.SerialIterator(train,
                                                  batch_size=1,
                                                  repeat=True,
                                                  shuffle=True)

    updater = chainer.training.StandardUpdater(train_iter, optimizer)

    trainer = chainer.training.Trainer(updater, (1, 'epoch'))

    @extension.make_extension(
        trigger=(1, 'iteration'), priority=extension.PRIORITY_WRITER)
    def rank_reporter(trainer):
        tmp = xp.asarray(comm.rank, dtype=np.float32)
        if use_chainer_variable:
            tmp = chainer.Variable(tmp)
        trainer.observation['rank'] = tmp

    @extension.make_extension(
        trigger=(communicate_interval, 'iteration'),
        priority=extension.PRIORITY_READER)
    def aggregated_rank_checker(trainer):
        actual = trainer.observation['rank-aggregated']
        if use_chainer_variable:
            actual = actual.data
        expected = (comm.size - 1) / 2
        chainer.testing.assert_allclose(actual, expected)

    trainer.extend(rank_reporter)
    trainer.extend(ObservationAggregator(
        comm, 'rank', 'rank-aggregated',
        comm_trigger=(communicate_interval, 'iteration')))
    trainer.extend(aggregated_rank_checker)

    trainer.run()
