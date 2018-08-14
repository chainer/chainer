import chainer
import chainer.testing
import chainer.testing.attr
import unittest

import chainermn


class ExampleModel(chainer.Chain):
    def __init__(self, n_in=3, n_units=5, n_out=2):
        super(ExampleModel, self).__init__(
            l1=chainer.links.Linear(n_in, n_units, nobias=True),
            bn1=chainer.links.BatchNormalization(n_units),
            l2=chainer.links.Linear(n_units, n_units, nobias=True),
            bn2=chainer.links.BatchNormalization(n_units),
            l3=chainer.links.Linear(n_units, n_out),
        )


class TestAllreducePersistent(unittest.TestCase):

    def _test(self, comm, model):
        rank = comm.rank
        model.bn1.avg_mean.fill(rank * 1)
        model.bn2.avg_mean.fill(rank * 2)
        model.bn1.avg_var.fill(rank * 3)
        model.bn2.avg_var.fill(rank * 4)

        allreduce_persistent = \
            chainermn.extensions.AllreducePersistent(model, comm)
        allreduce_persistent()

        avg_rank = (comm.size - 1) / 2.0
        chainer.testing.assert_allclose(model.bn1.avg_mean, avg_rank * 1)
        chainer.testing.assert_allclose(model.bn2.avg_mean, avg_rank * 2)
        chainer.testing.assert_allclose(model.bn1.avg_var, avg_rank * 3)
        chainer.testing.assert_allclose(model.bn2.avg_var, avg_rank * 4)

    def test_allreduce_persistent_cpu(self):
        comm = chainermn.create_communicator('naive')
        self._test(comm, ExampleModel())

    @chainer.testing.attr.gpu
    def test_allreduce_persistent_gpu(self):
        comm = chainermn.create_communicator('hierarchical')
        device = comm.intra_rank
        chainer.cuda.get_device_from_id(device).use()

        model = ExampleModel()
        model.to_gpu()
        self._test(comm, model)
