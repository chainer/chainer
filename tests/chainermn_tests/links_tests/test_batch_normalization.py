import chainer
import chainer.testing
import chainer.utils
import mpi4py.MPI
import numpy
import unittest

import chainermn
from chainermn.communicators.naive_communicator import NaiveCommunicator
import chainermn.links


class ModelNormalBN(chainer.Chain):
    def __init__(self, n_in=3, n_units=3, n_out=2):
        super(ModelNormalBN, self).__init__()
        with self.init_scope():
            self.l1 = chainer.links.Linear(n_in, n_units, nobias=True)
            self.bn1 = chainer.links.BatchNormalization(n_units)
            self.l2 = chainer.links.Linear(n_in, n_units, nobias=True)
            self.bn2 = chainer.links.BatchNormalization(n_units)
            self.l3 = chainer.links.Linear(n_in, n_out)
        self.train = True

    def __call__(self, x):
        h = chainer.functions.relu(self.bn1(self.l1(x)))
        h = chainer.functions.relu(self.bn2(self.l2(h)))
        return self.l3(h)


class ModelDistributedBN(chainer.Chain):
    def __init__(self, comm, n_in=3, n_units=3, n_out=2):
        super(ModelDistributedBN, self).__init__()
        with self.init_scope():
            self.l1 = chainer.links.Linear(n_in, n_units, nobias=True)
            self.bn1 = chainermn.links.MultiNodeBatchNormalization(
                n_units, comm)
            self.l2 = chainer.links.Linear(n_in, n_units, nobias=True)
            self.bn2 = chainermn.links.MultiNodeBatchNormalization(
                n_units, comm)
            self.l3 = chainer.links.Linear(n_in, n_out)
        self.train = True

    def __call__(self, x):
        h = chainer.functions.relu(self.bn1(self.l1(x)))
        h = chainer.functions.relu(self.bn2(self.l2(h)))
        return self.l3(h)


class TestMultiNodeBatchNormalization(unittest.TestCase):

    def setUp(self):
        self.mpi_comm = mpi4py.MPI.COMM_WORLD
        self.communicator = NaiveCommunicator(self.mpi_comm)

    def test_version_check(self):
        if chainer.__version__.startswith('1.'):
            with self.assertRaises(RuntimeError):
                chainermn.links.MultiNodeBatchNormalization(
                    3, self.communicator)
        else:
            # Expecting no exceptions
            chainermn.links.MultiNodeBatchNormalization(
                3, self.communicator)

    def test_multi_node_bn(self):
        """Tests correctness of MultiNodeBatchNormalization.

        This test verifies MultiNodeBatchNormalization by comparing
        the following four configurations.
        (1) Single worker, normal BatchNormalization
        (2) Multiple workers, normal BatchNormalization
        (3) Single worker, MultiNodeBatchNormalization
        (4) Multiple workers, MultiNodeBatchNormalization

        Single worker: only using the result of worker 0, which uses the whole
            batch.
        Multiple workers: Each worker uses the 1/n part of the whole batch,
            where n is the number of nodes, and gradient is aggregated.

        This test conducts the forward and backward computation once for the
        deterministic model parameters and an input batch, and checks the
        gradients of parameters.

        The purpose of MultiNodeBatchNormalization is to make the results of
        (4) to be exactly same as (1). Therefore, the essential part is to
        check that the results of (1) and (4) are the same. The results of (3)
        should also be also same as them. In contrast, the results of (2) is
        not necessarily always same as them, and we can expect that it is
        almost always different. Therefore, we also check that the results of
        (2) is different from them, to see that this test working correctly.
        """

        comm = self.communicator

        local_batchsize = 10
        global_batchsize = 10 * comm.size
        ndim = 3
        numpy.random.seed(71)
        x = numpy.random.random(
            (global_batchsize, ndim)).astype(numpy.float32)
        y = numpy.random.randint(
            0, 1, size=global_batchsize, dtype=numpy.int32)
        x_local = comm.mpi_comm.scatter(
            x.reshape(comm.size, local_batchsize, ndim))
        y_local = comm.mpi_comm.scatter(
            y.reshape(comm.size, local_batchsize))

        cls = chainer.links.Classifier
        m1 = cls(ModelNormalBN())           # Single worker
        m2 = cls(ModelNormalBN())           # Multi worker, Ghost BN
        m3 = cls(ModelDistributedBN(comm))  # Single worker, MNBN
        m4 = cls(ModelDistributedBN(comm))  # Multi worker, MNBN
        # NOTE: m1, m3 and m4 should behave in the same way.
        # m2 may be different.

        m2.copyparams(m1)
        m3.copyparams(m1)
        m4.copyparams(m1)

        l1 = m1(x, y)
        m1.cleargrads()
        l1.backward()

        l2 = m2(x_local, y_local)
        m2.cleargrads()
        l2.backward()
        comm.allreduce_grad(m2)

        l3 = m3(x, y)
        m3.cleargrads()
        l3.backward()

        l4 = m4(x_local, y_local)
        m4.cleargrads()
        l4.backward()
        comm.allreduce_grad(m4)

        if comm.rank == 0:
            for p1, p2, p3, p4 in zip(
                    sorted(m1.namedparams()),
                    sorted(m2.namedparams()),
                    sorted(m3.namedparams()),
                    sorted(m4.namedparams())):
                name = p1[0]
                assert(p2[0] == name)
                assert(p3[0] == name)
                assert(p4[0] == name)

                chainer.testing.assert_allclose(p1[1].grad, p3[1].grad)
                chainer.testing.assert_allclose(p1[1].grad, p4[1].grad)

                # This is to see that this test is valid.
                if comm.size >= 2:
                    self.assert_not_allclose(p1[1].grad, p2[1].grad)

    def assert_not_allclose(self, x, y, atol=1e-5, rtol=1e-4, verbose=True):
        x = chainer.cuda.to_cpu(chainer.utils.force_array(x))
        y = chainer.cuda.to_cpu(chainer.utils.force_array(y))

        with self.assertRaises(AssertionError):
            numpy.testing.assert_allclose(
                x, y, atol=atol, rtol=rtol, verbose=verbose)
