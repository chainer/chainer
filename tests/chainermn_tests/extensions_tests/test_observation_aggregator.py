from __future__ import division

import pytest

import numpy as np

import chainer
import chainer.testing
from chainer.training import extension
from chainer.backend import cuda
import chainermn
from chainermn.testing import get_device
from chainermn.extensions import ObservationAggregator
import chainerx


class DummyChain(chainer.Chain):

    def __init__(self):
        super(DummyChain, self).__init__()
        with self.init_scope():
            self.l = chainer.links.Linear(None, 1)

    def forward(self, x):
        return chainer.functions.sum(self.l(x))


@pytest.mark.parametrize('use_chainer_variable', [False, True])
@pytest.mark.parametrize('communicate_interval', [1, 2])
@pytest.mark.parametrize('xp', [chainerx, np])
def test_observation_aggregator_cpu(use_chainer_variable,
                                    communicate_interval,
                                    xp):
    communicator = chainermn.create_communicator('naive')
    run_test_observation_aggregator(communicator, xp,
                                    use_chainer_variable,
                                    communicate_interval,
                                    use_gpu=False)


@pytest.mark.parametrize('use_chainer_variable', [False, True])
@pytest.mark.parametrize('communicate_interval', [1, 2])
@chainer.testing.attr.gpu
def test_observation_aggregator_gpu_chainerx(use_chainer_variable,
                                             communicate_interval):
    xp = chainerx
    communicator = chainermn.create_communicator('pure_nccl')
    device = get_device(communicator.intra_rank, True)
    with chainerx.using_device(device.device):
        if use_chainer_variable:
            run_test_observation_aggregator(communicator, xp,
                                            use_chainer_variable,
                                            communicate_interval,
                                            use_gpu=True)
        else:
            with pytest.raises(ValueError):
                run_test_observation_aggregator(communicator, xp,
                                                use_chainer_variable,
                                                communicate_interval,
                                                use_gpu=True)


@pytest.mark.parametrize('use_chainer_variable', [True, False])
@pytest.mark.parametrize('communicate_interval', [1, 2])
@chainer.testing.attr.gpu
def test_observation_aggregator_gpu_cupy(use_chainer_variable,
                                         communicate_interval):
    communicator = chainermn.create_communicator('pure_nccl')

    run_test_observation_aggregator(communicator, cuda.cupy,
                                    use_chainer_variable,
                                    communicate_interval,
                                    use_gpu=True)


def run_test_observation_aggregator(comm, xp,
                                    use_chainer_variable,
                                    communicate_interval,
                                    use_gpu):
    model = DummyChain()

    if use_gpu:
        # Use CuPy's Device class to force call cudaSetDevice()
        chainer.cuda.get_device_from_id(comm.intra_rank).use()

    device = get_device(comm.intra_rank if use_gpu else None, xp == chainerx)

    if xp == chainerx:
        train = xp.array(np.random.rand(10, 1).astype(np.float32))
    else:
        train = xp.random.rand(10, 1).astype(np.float32)

    model.to_device(device)

    train_iter = chainer.iterators.SerialIterator(train,
                                                  batch_size=1,
                                                  repeat=True,
                                                  shuffle=True)

    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.Adam(), comm)
    optimizer.setup(model)

    updater = chainer.training.StandardUpdater(train_iter, optimizer,
                                               device=device)

    trainer = chainer.training.Trainer(updater, (1, 'epoch'))

    @extension.make_extension(
        trigger=(1, 'iteration'), priority=extension.PRIORITY_WRITER)
    def rank_reporter(trainer_):
        tmp = xp.asarray(comm.rank, dtype=np.float32)
        if use_chainer_variable:
            tmp = chainer.Variable(tmp)
        trainer_.observation['rank'] = tmp

    @extension.make_extension(
        trigger=(communicate_interval, 'iteration'),
        priority=extension.PRIORITY_READER)
    def aggregated_rank_checker(trainer_):
        actual = trainer_.observation['rank-aggregated']
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
