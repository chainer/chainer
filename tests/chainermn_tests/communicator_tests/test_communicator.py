import mpi4py.MPI
import numpy as np
import pytest
import unittest

import chainer
import chainer.cuda
import chainer.links
import chainer.testing
import chainer.testing.attr
import chainermn
from chainermn.communicators import _communication_utility
from chainermn.communicators.flat_communicator \
    import FlatCommunicator
from chainermn.communicators.hierarchical_communicator \
    import HierarchicalCommunicator
from chainermn.communicators.naive_communicator \
    import NaiveCommunicator
from chainermn.communicators.non_cuda_aware_communicator \
    import NonCudaAwareCommunicator
from chainermn.communicators.pure_nccl_communicator \
    import PureNcclCommunicator
from chainermn.communicators.single_node_communicator \
    import SingleNodeCommunicator
from chainermn.communicators.two_dimensional_communicator \
    import TwoDimensionalCommunicator
from chainermn import nccl


class ExampleModel(chainer.Chain):

    def __init__(self, dtype=None):
        if dtype is not None:
            self.dtype = dtype
        super(ExampleModel, self).__init__()
        with self.init_scope():
            self.a = chainer.links.Linear(2, 3)
            self.b = chainer.links.Linear(3, 4)
            self.c = chainer.links.Linear(4, 5)


class Param(object):
    def __init__(self, param):
        self.gpu = False
        self.nccl1 = False
        self.model_dtype = None
        self.allreduce_grad_dtype = None
        self.__dict__.update(param)


cpu_params = [Param(p) for p in [
    {
        'communicator_class': NaiveCommunicator,
        'multi_node': True,
    }]]
gpu_params = [Param(p) for p in [
    {
        'communicator_class': NaiveCommunicator,
        'multi_node': True,
    }, {
        'communicator_class': FlatCommunicator,
        'multi_node': True,
    }, {
        'communicator_class': HierarchicalCommunicator,
        'multi_node': True,
    }, {
        'communicator_class': TwoDimensionalCommunicator,
        'multi_node': True,
    }, {
        'communicator_class': SingleNodeCommunicator,
        'multi_node': False,
    }, {
        'communicator_class': NonCudaAwareCommunicator,
        'multi_node': True,
    }, {
        'communicator_class': PureNcclCommunicator,
        'multi_node': True,
        'nccl1': False,
    }, {
        'communicator_class': PureNcclCommunicator,
        'multi_node': True,
        'nccl1': False,
        'allreduce_grad_dtype': np.float16,
    }, {
        'communicator_class': PureNcclCommunicator,
        'multi_node': True,
        'nccl1': False,
        'model_dtype': np.float16,
        'allreduce_grad_dtype': np.float16,
    }, {
        'communicator_class': PureNcclCommunicator,
        'multi_node': True,
        'nccl1': False,
        'model_dtype': np.float64,
        'allreduce_grad_dtype': np.float64,
    }]]

mpi_comm = mpi4py.MPI.COMM_WORLD


def create_communicator(param, use_gpu):
    if not param.multi_node:
        ranks = _communication_utility.init_ranks(mpi_comm)
        inter_size = ranks[4]
        if inter_size > 1:
            pytest.skip('This test is for single node only')

    if use_gpu and not param.nccl1 and nccl.get_version() < 2000:
        pytest.skip('This test requires NCCL version >= 2.0')

    if param.allreduce_grad_dtype is not None:
        dtype = param.allreduce_grad_dtype
        communicator = \
            param.communicator_class(mpi_comm,
                                     allreduce_grad_dtype=dtype)
    else:
        communicator = param.communicator_class(mpi_comm)

    if use_gpu:
        chainer.cuda.get_device_from_id(communicator.intra_rank).use()

    return communicator


def check_send_and_recv(communicator, *shape):
    if communicator.size < 2:
        pytest.skip("This test is for multiple nodes")

    if communicator.rank > 0:
        rank_prev = (communicator.rank - 1) % communicator.size
        data_recv = communicator.recv(source=rank_prev, tag=0)
        chainer.testing.assert_allclose(
            data_recv, rank_prev * np.ones((shape)))

    if communicator.rank < communicator.size - 1:
        rank_next = (communicator.rank + 1) % communicator.size
        data_send = communicator.rank * \
            np.ones((shape)).astype(np.float32)
        communicator.send(data_send, dest=rank_next, tag=0)


def check_send_and_recv_tuple(communicator, data):
    if communicator.size < 2:
        pytest.skip("This test is for multiple nodes")

    if communicator.rank > 0:
        rank_prev = (communicator.rank - 1) % communicator.size
        data_recv = communicator.recv(source=rank_prev, tag=0)
        for array0, array1 in zip(data, data_recv):
            chainer.testing.assert_allclose(array0, array1)

    if communicator.rank < communicator.size - 1:
        rank_next = (communicator.rank + 1) % communicator.size
        communicator.send(data, dest=rank_next, tag=0)


def check_bcast_data(communicator, model):
    model.a.W.data[:] = communicator.rank
    model.b.W.data[:] = communicator.rank + 1
    model.c.b.data[:] = communicator.rank + 2
    communicator.bcast_data(model)
    chainer.testing.assert_allclose(model.a.W.data, 0 * np.ones((3, 2)))
    chainer.testing.assert_allclose(model.b.W.data, 1 * np.ones((4, 3)))
    chainer.testing.assert_allclose(model.c.b.data, 2 * np.ones((5, )))


def check_allreduce_grad(communicator, model):
    # We need to repeat twice for regressions on lazy initialization of
    # sub communicators.
    for _ in range(2):
        model.a.W.grad[:] = communicator.rank
        model.b.W.grad[:] = communicator.rank + 1
        model.c.b.grad[:] = communicator.rank + 2

        communicator.allreduce_grad(model)
        base = (communicator.size - 1.0) / 2

        chainer.testing.assert_allclose(model.a.W.grad,
                                        (base + 0) * np.ones((3, 2)))
        chainer.testing.assert_allclose(model.b.W.grad,
                                        (base + 1) * np.ones((4, 3)))
        chainer.testing.assert_allclose(model.c.b.grad,
                                        (base + 2) * np.ones((5, )))


def check_allreduce_grad_empty(communicator, model):
    # We need to repeat twice for regressions on lazy initialization of
    # sub communicators.
    for _ in range(2):
        model.a.W.grad[:] = communicator.rank
        model.b.W.grad[:] = communicator.rank + 1
        model.c.b.grad = None

        communicator.allreduce_grad(model)
        base = (communicator.size - 1.0) / 2

        chainer.testing.assert_allclose(model.a.W.grad,
                                        (base + 0) * np.ones((3, 2)))
        chainer.testing.assert_allclose(model.b.W.grad,
                                        (base + 1) * np.ones((4, 3)))


def check_send_recv(param, use_gpu):
    communicator = create_communicator(param, use_gpu)

    assert mpi_comm.Get_rank() == communicator.rank
    assert mpi_comm.Get_size() == communicator.size

    check_send_and_recv(communicator, 50)
    check_send_and_recv(communicator, 50, 20)

    check_send_and_recv(communicator, 50, 20, 5)
    check_send_and_recv(communicator, 50, 20, 5, 3)

    data = [np.ones((50)).astype(np.float32)]
    check_send_and_recv_tuple(communicator, data)

    data = [
        np.ones((50)).astype(np.float32),
        np.ones((50, 20)).astype(np.float32),
        np.ones((50, 20, 5)).astype(np.float32)]
    check_send_and_recv_tuple(communicator, data)


def check_collective_communication(param, use_gpu):
    communicator = create_communicator(param, use_gpu)

    model = ExampleModel(param.model_dtype)
    if use_gpu:
        model.to_gpu()
    check_bcast_data(communicator, model)
    check_allreduce_grad(communicator, model)
    check_allreduce_grad_empty(communicator, model)
    # barrier() requires before destructor of PureNcclCommunicator
    # because communication may not be finished.
    communicator.mpi_comm.barrier()


# chainer.testing.parameterize is not available at functions
@pytest.mark.parametrize('param', cpu_params)
def test_communicator_cpu(param):
    check_send_recv(param, False)
    check_collective_communication(param, False)


@pytest.mark.parametrize('param', gpu_params)
@chainer.testing.attr.gpu
def test_communicator_gpu(param):
    check_send_recv(param, True)
    check_collective_communication(param, True)


class TestPureNcclCommunicator(unittest.TestCase):

    def setUp(self):
        if nccl.get_version() < 2000:
            pytest.skip('This test requires NCCL version >= 2.0')
        self.mpi_comm = mpi4py.MPI.COMM_WORLD

    @chainer.testing.attr.gpu
    def test_invalid_allreduce_grad_dtype(self):
        with self.assertRaises(ValueError):
            PureNcclCommunicator(self.mpi_comm, allreduce_grad_dtype=np.int32)


class TestDifferentDtype(unittest.TestCase):

    def setup(self, gpu):
        if gpu:
            self.communicator = chainermn.create_communicator('hierarchical')
            self.device = self.communicator.rank
            chainer.cuda.get_device_from_id(self.device).use()
        else:
            self.communicator = chainermn.create_communicator('naive')
            self.device = -1

        if self.communicator.size != 2:
            pytest.skip('This test is for two processes')

        # dtypes to be tested
        # DO NOT USE chainer.testing.parameterize
        # (because running order of generated test cases is not unique)
        self.dtypes = [np.int32, np.int64, np.float32, np.float64]

    def check_alltoall(self, xs):
        x = xs[self.communicator.rank]
        ys = self.communicator.alltoall(
            tuple([x for _ in range(self.communicator.size)]))
        for x, y in zip(xs, ys):
            chainer.testing.assert_allclose(x, y)

    def test_alltoall_cpu(self):
        self.setup(False)
        for dtype in self.dtypes:
            xs = np.arange(4 * self.communicator.size) \
                .reshape(self.communicator.size, 4) \
                .astype(dtype)
            xs = np.split(xs, self.communicator.size)
            self.check_alltoall(xs)

    @chainer.testing.attr.gpu
    def test_alltoall_gpu(self):
        self.setup(True)
        for dtype in self.dtypes:
            xs = np.arange(4 * self.communicator.size) \
                .reshape(self.communicator.size, 4) \
                .astype(dtype)
            xs = np.split(xs, self.communicator.size)
            xs = [chainer.cuda.to_gpu(x, device=self.device) for x in xs]
            self.check_alltoall(xs)

    def check_allgather(self, xs):
        x = xs[self.communicator.rank]
        ys = self.communicator.allgather(x)
        for x, y in zip(xs, ys):
            chainer.testing.assert_allclose(x, y)

    def test_allgather_cpu(self):
        self.setup(False)
        for dtype in self.dtypes:
            xs = np.arange(4 * self.communicator.size) \
                .reshape(self.communicator.size, 4) \
                .astype(dtype)
            xs = np.split(xs, self.communicator.size)
            self.check_allgather(xs)

    @chainer.testing.attr.gpu
    def test_allgather_gpu(self):
        self.setup(True)
        for dtype in self.dtypes:
            xs = np.arange(4 * self.communicator.size) \
                .reshape(self.communicator.size, 4) \
                .astype(dtype)
            xs = np.split(xs, self.communicator.size)
            xs = [chainer.cuda.to_gpu(x, device=self.device) for x in xs]
            self.check_allgather(xs)

    def check_bcast(self, x):
        if self.communicator.rank == 0:
            y = self.communicator.bcast(x, root=0)
        else:
            y = self.communicator.bcast(None, root=0)
        chainer.testing.assert_allclose(x, y)

    def test_bcast_cpu(self):
        self.setup(False)
        for dtype in self.dtypes:
            x = np.arange(4).astype(dtype)
            self.check_bcast(x)

    @chainer.testing.attr.gpu
    def test_bcast_gpu(self):
        self.setup(True)
        for dtype in self.dtypes:
            x = np.arange(4).astype(dtype)
            x = chainer.cuda.to_gpu(x, device=self.device)
            self.check_bcast(x)

    def check_gather(self, xs):
        x = xs[self.communicator.rank]
        ys = self.communicator.gather(x, root=0)
        if self.communicator.rank == 0:
            for x, y in zip(xs, ys):
                chainer.testing.assert_allclose(x, y)

    def test_gather_cpu(self):
        self.setup(False)
        for dtype in self.dtypes:
            xs = np.arange(4 * self.communicator.size) \
                .reshape(self.communicator.size, 4) \
                .astype(dtype)
            xs = np.split(xs, self.communicator.size)
            self.check_gather(xs)

    @chainer.testing.attr.gpu
    def test_gather_gpu(self):
        self.setup(True)
        for dtype in self.dtypes:
            xs = np.arange(4 * self.communicator.size) \
                .reshape(self.communicator.size, 4) \
                .astype(dtype)
            xs = np.split(xs, self.communicator.size)
            xs = [chainer.cuda.to_gpu(x, device=self.device) for x in xs]
            self.check_gather(xs)

    def check_scatter(self, xs):
        x = xs[self.communicator.rank]
        if self.communicator.rank == 0:
            y = self.communicator.scatter(xs, root=0)
        else:
            y = self.communicator.scatter(None, root=0)
        chainer.testing.assert_allclose(x, y)

    def test_scatter_cpu(self):
        self.setup(False)
        for dtype in self.dtypes:
            xs = np.arange(4 * self.communicator.size) \
                .reshape(self.communicator.size, 4) \
                .astype(dtype)
            xs = np.split(xs, self.communicator.size)
            self.check_scatter(xs)

    @chainer.testing.attr.gpu
    def test_scatter_gpu(self):
        self.setup(True)
        for dtype in self.dtypes:
            xs = np.arange(4 * self.communicator.size) \
                .reshape(self.communicator.size, 4) \
                .astype(dtype)
            xs = np.split(xs, self.communicator.size)
            xs = [chainer.cuda.to_gpu(x, device=self.device) for x in xs]
            self.check_scatter(xs)


class TestNonContiguousArray(unittest.TestCase):

    def setup(self, gpu):
        if gpu:
            self.communicator = chainermn.create_communicator('hierarchical')
            self.device = self.communicator.rank
            chainer.cuda.get_device_from_id(self.device).use()
        else:
            self.communicator = chainermn.create_communicator('naive')
            self.device = -1

        if self.communicator.size != 2:
            pytest.skip('This test is for two processes')

    def check_send(self):
        if self.communicator.rank == 0:
            x = np.arange(18).reshape(3, 3, 2).astype(np.float32)
            # slicing operator destruct both C-/Fortran-contiguousness
            self.communicator.send(x[:, 1, :], dest=1, tag=0)

        elif self.communicator.rank == 1:
            self.communicator.recv(source=0, tag=0)

    def test_send_cpu(self):
        self.setup(False)
        self.check_send()

    @chainer.testing.attr.gpu
    def test_send_gpu(self):
        self.setup(True)
        self.check_send()

    def check_alltoall(self):
        self.setup(False)
        x = np.arange(18).reshape(3, 3, 2).astype(np.float32)
        # slicing operator destruct both C-/Fortran-contiguousness
        x = x[:, 1, :]
        xs = (x, x)
        self.communicator.alltoall(xs)

    def test_alltoall_cpu(self):
        self.setup(False)
        self.check_alltoall()

    @chainer.testing.attr.gpu
    def test_alltoall_gpu(self):
        self.setup(True)
        self.check_alltoall()

    def check_allreduce(self):
        x = np.arange(18) + self.communicator.rank
        xs = x.astype(np.float32)
        xs = self.communicator.allreduce(xs)

        s = sum(range(self.communicator.size))

        y = np.arange(18) * self.communicator.size + s
        ys = y.astype(np.float32)
        chainer.testing.assert_allclose(ys, xs)

    def test_allreduce_cpu(self):
        self.setup(False)
        self.check_allreduce()

    @chainer.testing.attr.gpu
    def test_allreduce_gpu(self):
        self.setup(True)
        self.check_allreduce()
