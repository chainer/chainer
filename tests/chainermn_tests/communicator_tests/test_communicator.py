import mock
import mpi4py.MPI
import numpy as np
import pytest
import unittest

import chainer
import chainer.initializers
import chainer.links
import chainer.testing
import chainer.testing.attr
import chainermn
import chainerx
from chainermn.communicators import _communication_utility
from chainermn.communicators.flat_communicator \
    import FlatCommunicator
from chainermn.communicators.naive_communicator \
    import NaiveCommunicator
from chainermn.communicators.non_cuda_aware_communicator \
    import NonCudaAwareCommunicator
from chainermn.communicators.pure_nccl_communicator \
    import PureNcclCommunicator
from chainermn import nccl
import chainermn.testing


class ExampleModel(chainer.Chain):

    def __init__(self, dtype=None):
        W = None
        bias = None
        if dtype is not None:
            self.dtype = dtype
            W = chainer.initializers.Normal(dtype=self.dtype)
            bias = chainer.initializers.Zero(dtype=self.dtype)
        super(ExampleModel, self).__init__()
        with self.init_scope():
            self.a = chainer.links.Linear(2, 3, initialW=W, initial_bias=bias)
            self.b = chainer.links.Linear(3, 4, initialW=W, initial_bias=bias)
            self.c = chainer.links.Linear(None, 5, initialW=W,
                                          initial_bias=bias)


class ExampleMixedModel(chainer.Chain):
    def __init__(self):
        W16 = chainer.initializers.Normal(dtype=np.float16)
        W32 = chainer.initializers.Normal(dtype=np.float32)
        bias16 = chainer.initializers.Zero(dtype=np.float16)
        bias32 = chainer.initializers.Zero(dtype=np.float32)
        super(ExampleMixedModel, self).__init__()
        with self.init_scope():
            self.a = chainer.links.Linear(2, 3, initialW=W32,
                                          initial_bias=bias32)
            self.b = chainer.links.Linear(3, 4, initialW=W16,
                                          initial_bias=bias16)
            self.c = chainer.links.Linear(None, 5, initialW=W16,
                                          initial_bias=bias32)


class Param(object):
    def __init__(self, param):
        self.gpu = False
        self.nccl1 = False
        self.model_dtype = None
        self.allreduce_grad_dtype = None
        self.batched_copy = True
        self.global_dtype = None
        self.__dict__.update(param)

    def __repr__(self):
        import pprint
        return pprint.pformat(self.__dict__)


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
        'communicator_class': NaiveCommunicator,
        'model_dtype': np.float16,
        'multi_node': True,
    }, {
        'communicator_class': FlatCommunicator,
        'multi_node': True,
    }, {
        'communicator_class': FlatCommunicator,
        'model_dtype': np.float16,
        'multi_node': True,
    }, {
        'communicator_class': NonCudaAwareCommunicator,
        'multi_node': True,
    }, {
        'communicator_class': NonCudaAwareCommunicator,
        'model_dtype': np.float16,
        'multi_node': False,
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
        'model_dtype': np.float16,
        'allreduce_grad_dtype': np.float32,
    }, {
        'communicator_class': PureNcclCommunicator,
        'multi_node': True,
        'nccl1': False,
        'model_dtype': np.float32,
        'allreduce_grad_dtype': np.float32,
    }, {
        'communicator_class': PureNcclCommunicator,
        'multi_node': True,
        'nccl1': False,
        'model_dtype': np.float32,
        'allreduce_grad_dtype': np.float16,
    }]]


gpu_mixed_dtype_params = [Param(p) for p in [
    {
        'communicator_class': NonCudaAwareCommunicator,
        'multi_node': True,
    }, {
        'communicator_class': NaiveCommunicator,
        'multi_node': True,
    }, {
        'communicator_class': FlatCommunicator,
        'multi_node': True,
    }
]]
for global_dtype in [np.float32, np.float16, chainer.mixed16, None]:
    for allreduce_dtype in [np.float32, np.float16, None]:
        if global_dtype is None and allreduce_dtype is None:
            continue
        for batched_copy in [True, False]:
            gpu_mixed_dtype_params.append(Param({
                'communicator_class': PureNcclCommunicator,
                'multi_node': True,
                'global_dtype': global_dtype,
                'allreduce_grad_dtype': allreduce_dtype,
                'batched_copy': batched_copy,
            }))


mpi_comm = mpi4py.MPI.COMM_WORLD


def create_communicator(param, use_gpu, use_chx):
    if not param.multi_node:
        ranks = _communication_utility.init_ranks(mpi_comm)
        inter_size = ranks[4]
        if inter_size > 1:
            pytest.skip('This test is for single node only')

    if use_gpu and not param.nccl1 and nccl.get_build_version() < 2000:
        pytest.skip('This test requires NCCL version >= 2.0')

    communicator = param.communicator_class(mpi_comm)
    communicator.set_config('batched_copy', param.batched_copy)
    value = communicator.get_config('batched_copy')
    assert param.batched_copy == value

    with pytest.raises(ValueError):
        communicator.set_config('blah blah blah')

    if param.communicator_class is PureNcclCommunicator:
        communicator.set_config('allreduce_grad_dtype',
                                param.allreduce_grad_dtype)
        value = communicator.get_config('allreduce_grad_dtype')
        assert param.allreduce_grad_dtype == value

    if use_gpu:
        chainermn.testing.get_device(communicator.intra_rank, use_chx).use()

    return communicator


def check_send_and_recv(communicator, *shape):
    if communicator.size < 2:
        pytest.skip('This test is for multiple nodes')

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
        pytest.skip('This test is for multiple nodes')

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


def check_multi_node_mean_grad(communicator, model):
    # We need to repeat twice for regressions on lazy initialization of
    # sub communicators.

    for _ in range(2):
        model.a.W.grad[:] = communicator.rank
        model.b.W.grad[:] = communicator.rank + 1
        model.c.b.grad[:] = communicator.rank + 2

        communicator.multi_node_mean_grad(model)
        base = (communicator.size - 1.0) / 2

        chainer.testing.assert_allclose(model.a.W.grad,
                                        (base + 0) * np.ones((3, 2)))
        chainer.testing.assert_allclose(model.b.W.grad,
                                        (base + 1) * np.ones((4, 3)))
        chainer.testing.assert_allclose(model.c.b.grad,
                                        (base + 2) * np.ones((5, )))


def check_multi_node_mean_grad_empty(communicator, model):
    # We need to repeat twice for regressions on lazy initialization of
    # sub communicators.
    for _ in range(2):
        model.a.W.grad[:] = communicator.rank
        model.b.W.grad[:] = communicator.rank + 1
        model.c.b.grad = None

        communicator.multi_node_mean_grad(model)
        base = (communicator.size - 1.0) / 2

        chainer.testing.assert_allclose(model.a.W.grad,
                                        (base + 0) * np.ones((3, 2)))
        chainer.testing.assert_allclose(model.b.W.grad,
                                        (base + 1) * np.ones((4, 3)))


def check_multi_node_mean_grad_empty_half(communicator, model):
    # We need to repeat twice for regressions on lazy initialization of
    # sub communicators.

    for _ in range(2):
        model.a.W.data[:] = communicator.rank
        model.b.W.data[:] = communicator.rank + 1
        model.c.b.data[:] = communicator.rank + 2

        model.a.W.grad[:] = communicator.rank
        model.b.W.grad[:] = communicator.rank + 1
        if communicator.rank % 2 == 0:
            model.c.b.grad[:] = communicator.rank + 2
        else:
            model.c.b.grad = None

        communicator.multi_node_mean_grad(model, zero_fill=True)
        base = (communicator.size - 1.0) / 2

        chainer.testing.assert_allclose(model.a.W.grad,
                                        (base + 0) * np.ones((3, 2)))
        chainer.testing.assert_allclose(model.b.W.grad,
                                        (base + 1) * np.ones((4, 3)))

        v = 0.0
        for i in range(communicator.size):
            if i % 2 == 0:
                v += i + 2
        v /= communicator.size
        chainer.testing.assert_allclose(model.c.b.grad,
                                        v * np.ones((5, )))


def check_send_recv(param, use_gpu, use_chx=False):
    communicator = create_communicator(param, use_gpu, use_chx)

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

    communicator.finalize()


def check_multi_node_mean_grad_mixed_dtype(param, model, use_gpu, use_chx):
    # Checks the actual allreduce communication is performed
    # in the correct data type (FP16 or FP32)
    comm_class = param.communicator_class

    if not param.multi_node:
        ranks = _communication_utility.init_ranks(mpi_comm)
        inter_size = ranks[4]
        if inter_size > 1:
            pytest.skip('This test is for single node only')

    communicator = comm_class(mpi_comm)
    communicator.set_config('batched_copy', param.batched_copy)

    if comm_class is PureNcclCommunicator:
        communicator.set_config('allreduce_grad_dtype',
                                param.allreduce_grad_dtype)
        value = communicator.get_config('allreduce_grad_dtype')
        assert param.allreduce_grad_dtype == value
        value = communicator.allreduce_grad_dtype
        assert param.allreduce_grad_dtype == value

    mpi_comm.barrier()

    # answer type: see the document of `create_communicator`
    global_dtype = param.global_dtype
    allreduce_dtype = param.allreduce_grad_dtype

    # assert test configuration.
    assert chainer.get_dtype() == global_dtype

    answer_dtype = None
    if allreduce_dtype == np.float16:
        answer_dtype = np.float16
    elif allreduce_dtype == np.float32:
        answer_dtype = np.float32
    else:
        if global_dtype == np.float32:
            answer_dtype = np.float32
        else:
            answer_dtype = np.float16

    if use_gpu:
        device = chainermn.testing.get_device(communicator.intra_rank,
                                              use_chainerx=use_chx)
        model.to_device(device)

    model.a.W.grad[:] = communicator.rank
    model.b.W.grad[:] = communicator.rank + 1
    model.c.b.grad[:] = communicator.rank + 2

    if isinstance(communicator, PureNcclCommunicator):
        communicator._init_comms()
        with mock.patch.object(communicator, 'nccl_comm',
                               wraps=communicator.nccl_comm) as mc:
            answer_dtype = _communication_utility._get_nccl_type_id(
                answer_dtype)

            communicator.multi_node_mean_grad(model)

            # dtype that was used in the actual communication,
            # which is nccl_comm.allReduce
            call_args = mc.allReduce.call_args[0]
            actual_dtype = call_args[3]
            assert answer_dtype == actual_dtype
    else:
        # For other MPI-based communicators,
        # all communication should happen in FP32 as of now, so
        # here we just check the results are correct for
        # 16-32 mixed models.
        communicator.multi_node_mean_grad(model)

    base = (communicator.size - 1.0) / 2
    chainer.testing.assert_allclose(model.a.W.grad,
                                    (base + 0) * np.ones((3, 2)))
    chainer.testing.assert_allclose(model.b.W.grad,
                                    (base + 1) * np.ones((4, 3)))

    mpi_comm.barrier()
    communicator.finalize()


def check_collective_communication(param, use_gpu, use_chx):
    communicator = create_communicator(param, use_gpu, use_chx)
    mpi_comm.barrier()

    model = ExampleModel(param.model_dtype)
    if use_gpu:
        device = chainermn.testing.get_device(communicator.intra_rank, use_chx)
    else:
        device = chainermn.testing.get_device(use_chainerx=use_chx)

    model.to_device(device)
    check_bcast_data(communicator, model)

    model = ExampleModel(param.model_dtype)
    model.to_device(device)
    check_multi_node_mean_grad(communicator, model)

    model = ExampleModel(param.model_dtype)
    model.to_device(device)
    check_multi_node_mean_grad_empty(communicator, model)

    model = ExampleModel(param.model_dtype)
    model.to_device(device)
    check_multi_node_mean_grad_empty_half(communicator, model)

    # Check allreduce debug mode
    model = ExampleModel()
    model.to_device(device)
    # The example model includes some nan parameters so the debug mode
    # must detect it.
    chainer.set_debug(True)
    with pytest.raises(ValueError, match=r'.* diverged .*'):
        check_multi_node_mean_grad(communicator, model)
    chainer.set_debug(False)

    # barrier() requires before destructor of PureNcclCommunicator
    # because communication may not be finished.
    mpi_comm.barrier()
    communicator.finalize()


# chainer.testing.parameterize is not available at functions
@pytest.mark.parametrize('param', cpu_params)
@pytest.mark.parametrize('use_chx', [True, False])
def test_communicator_cpu(param, use_chx):
    check_send_recv(param, False, use_chx)
    check_collective_communication(param, False, use_chx)


@pytest.mark.parametrize('param', gpu_params)
@pytest.mark.parametrize('use_chx', [True, False])
@chainer.testing.attr.gpu
def test_communicator_gpu(param, use_chx):
    check_send_recv(param, True)
    check_collective_communication(param, True, use_chx)


@pytest.mark.parametrize('param', gpu_mixed_dtype_params)
@pytest.mark.parametrize('use_chx', [True, False])
@chainer.testing.attr.gpu
def test_mixed_dtype_communicator_gpu(param, use_chx):
    model = ExampleMixedModel()
    with chainer.using_config('dtype', param.global_dtype):
        check_multi_node_mean_grad_mixed_dtype(param, model, True, use_chx)


class TestPureNcclCommunicator(unittest.TestCase):

    def setUp(self):
        if nccl.get_build_version() < 2000:
            pytest.skip('This test requires NCCL version >= 2.0')
        self.mpi_comm = mpi4py.MPI.COMM_WORLD

    @chainer.testing.attr.gpu
    def test_invalid_allreduce_grad_dtype(self):
        with self.assertRaises(ValueError):
            comm = PureNcclCommunicator(self.mpi_comm)
            comm.set_config('allreduce_grad_dtype', np.int32)

    @chainer.testing.attr.gpu
    def test_finalize(self):
        communicator = PureNcclCommunicator(self.mpi_comm)
        communicator._init_comms()
        communicator.finalize()
        self.assertIsNone(communicator.nccl_comm)


class TestNonCudaAwareCommunicator(unittest.TestCase):

    def setUp(self):
        if nccl.get_build_version() < 2000:
            pytest.skip('This test requires NCCL version >= 2.0')
        self.mpi_comm = mpi4py.MPI.COMM_WORLD

    @chainer.testing.attr.gpu
    def test_finalize(self):
        communicator = NonCudaAwareCommunicator(self.mpi_comm)
        communicator._init_comms()
        communicator.finalize()
        self.assertIsNone(communicator.intra_nccl_comm)


class TestDifferentDtype(unittest.TestCase):

    def setup(self, gpu):
        if gpu:
            self.communicator = chainermn.create_communicator('flat')
            self.device = self.communicator.intra_rank
            chainer.cuda.get_device_from_id(self.device).use()
        else:
            self.communicator = chainermn.create_communicator('naive')
            self.device = -1

        if self.communicator.size != 2:
            pytest.skip('This test is for two processes')

        # dtypes to be tested
        # DO NOT USE chainer.testing.parameterize
        # (because running order of generated test cases is not deterministic)
        self.dtypes = [np.int32, np.int64, np.float32, np.float64]

    def teardown(self):
        if self.communicator:
            self.communicator.finalize()

    def check_send_recv(self, x):
        if self.communicator.rank == 0:
            self.communicator.send(x, dest=1, tag=0)
            y = x

        elif self.communicator.rank == 1:
            y = self.communicator.recv(source=0, tag=0)

        chainer.testing.assert_allclose(y, x)

    def test_send_recv_cpu(self):
        self.setup(False)
        for dtype in self.dtypes:
            x = np.arange(18).astype(dtype)
            self.check_send_recv(x)

            x = np.array(1).astype(dtype)
            self.check_send_recv(x)
        self.teardown()

    @chainer.testing.attr.gpu
    def test_send_recv_gpu(self):
        self.setup(True)
        for dtype in self.dtypes:
            x = np.arange(18).astype(dtype)
            x = chainer.cuda.to_gpu(x, device=self.device)
            self.check_send_recv(x)
        self.teardown()

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

            xs = [np.array(1).astype(dtype)] * 4
            self.check_alltoall(xs)
        self.teardown()

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

            xs = [np.array(1).astype(dtype)] * 4
            xs = [chainer.cuda.to_gpu(x, device=self.device) for x in xs]
            self.check_alltoall(xs)
        self.teardown()

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

            x = np.array(1).astype(dtype)
            ys = self.communicator.allgather(x)
            for y in ys:
                chainer.testing.assert_allclose(x, y)
        self.teardown()

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

            x = np.array(1).astype(dtype)
            x = chainer.cuda.to_gpu(x, device=self.device)
            ys = self.communicator.allgather(x)
            for y in ys:
                chainer.testing.assert_allclose(x, y)
        self.teardown()

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

            x = np.array(42).astype(dtype)
            y = self.communicator.bcast(x)
            chainer.testing.assert_allclose(x, y)
        self.teardown()

    @chainer.testing.attr.gpu
    def test_bcast_gpu(self):
        self.setup(True)
        for dtype in self.dtypes:
            x = np.arange(4).astype(dtype)
            x = chainer.cuda.to_gpu(x, device=self.device)
            self.check_bcast(x)

            x = np.array(42).astype(dtype)
            x = chainer.cuda.to_gpu(x, device=self.device)
            y = self.communicator.bcast(x)
            chainer.testing.assert_allclose(x, y)
        self.teardown()

    def check_gather(self, xs, x1, ans):
        x = xs[self.communicator.rank]
        ys = self.communicator.gather(x, root=0)
        if self.communicator.rank == 0:
            for x, y in zip(xs, ys):
                chainer.testing.assert_allclose(x, y)

        ys = self.communicator.gather(x1, root=0)
        if self.communicator.rank == 0:
            for a, y in zip(ans, ys):
                chainer.testing.assert_allclose(a, y)

    def test_gather_cpu(self):
        self.setup(False)
        for dtype in self.dtypes:
            xs = np.arange(4 * self.communicator.size) \
                .reshape(self.communicator.size, 4) \
                .astype(dtype)
            xs = np.split(xs, self.communicator.size)

            x = np.array(self.communicator.rank).astype(dtype)
            ans = np.arange(self.communicator.size, dtype=dtype)
            self.check_gather(xs, x, ans)
        self.teardown()

    @chainer.testing.attr.gpu
    def test_gather_gpu(self):
        self.setup(True)
        for dtype in self.dtypes:
            xs = np.arange(4 * self.communicator.size) \
                .reshape(self.communicator.size, 4) \
                .astype(dtype)
            xs = np.split(xs, self.communicator.size)
            xs = [chainer.cuda.to_gpu(x, device=self.device) for x in xs]

            x = np.array(self.communicator.rank).astype(dtype)
            x = chainer.cuda.to_gpu(x, device=self.device)
            ans = np.arange(self.communicator.size, dtype=dtype)
            self.check_gather(xs, x, ans)
        self.teardown()

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

            x = np.array(42).astype(dtype)
            xs = [x] * self.communicator.size
            y = self.communicator.scatter(xs)
            chainer.testing.assert_allclose(x, y)
        self.teardown()

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

            x = np.array(42).astype(dtype)
            x = chainer.cuda.to_gpu(x, device=self.device)
            xs = [x] * self.communicator.size
            y = self.communicator.scatter(xs)
            chainer.testing.assert_allclose(x, y)
        self.teardown()

    def check_allreduce(self, x, dtype, n):
        x = self.communicator.allreduce(x)

        s = sum(range(self.communicator.size))

        y = np.arange(n) * self.communicator.size + s
        y = y.astype(dtype)
        chainer.testing.assert_allclose(y, x)

    def test_allreduce_cpu(self):
        self.setup(False)
        for dtype in self.dtypes:
            for n in [1, 18, 32]:
                x = np.arange(n) + self.communicator.rank
                x = x.astype(dtype)
                self.check_allreduce(x, dtype, n)

            x = np.array(1).astype(dtype)
            y = self.communicator.allreduce(x)
            a = x * self.communicator.size
            chainer.testing.assert_allclose(a, y)
        self.teardown()

    @chainer.testing.attr.gpu
    def test_allreduce_gpu(self):
        self.setup(True)
        for dtype in self.dtypes:
            x = np.arange(18) + self.communicator.rank
            x = x.astype(dtype)
            x = chainer.cuda.to_gpu(x, device=self.device)
            self.check_allreduce(x, dtype, 18)

            x = np.array(1).astype(dtype)
            y = self.communicator.allreduce(x)
            a = x * self.communicator.size
            chainer.testing.assert_allclose(a, y)
        self.teardown()


class TestNonContiguousArray(unittest.TestCase):

    def setup(self, gpu):
        if gpu:
            self.communicator = chainermn.create_communicator('flat')
            self.device = self.communicator.intra_rank
            chainer.cuda.get_device_from_id(self.device).use()
        else:
            self.communicator = chainermn.create_communicator('naive')
            self.device = -1

        if self.communicator.size != 2:
            pytest.skip('This test is for two processes')

    def teardown(self):
        if self.communicator:
            self.communicator.finalize()

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
        self.teardown()

    @chainer.testing.attr.gpu
    def test_send_gpu(self):
        self.setup(True)
        self.check_send()
        self.teardown()

    def check_alltoall(self):
        self.setup(False)
        x = np.arange(18).reshape(3, 3, 2).astype(np.float32)
        # slicing operator destruct both C-/Fortran-contiguousness
        x = x[:, 1, :]
        xs = (x, x)
        self.communicator.alltoall(xs)
        self.teardown()

    def test_alltoall_cpu(self):
        self.setup(False)
        self.check_alltoall()
        self.teardown()

    @chainer.testing.attr.gpu
    def test_alltoall_gpu(self):
        self.setup(True)
        self.check_alltoall()
        self.teardown()


class TestMpiCommunicatorBase(unittest.TestCase):

    def setup(self):
        self.communicator = chainermn.create_communicator('naive')

        if self.communicator.size != 2:
            pytest.skip('This test is for two processes')

    def teardown(self):
        if self.communicator:
            self.communicator.finalize()

    def check_send_recv_obj(self, x, tag=0,
                            use_any_recv=True, use_status=False):
        if self.communicator.rank == 0:
            self.communicator.send_obj(x, dest=1, tag=tag)
            y = x

        elif self.communicator.rank == 1:
            status = None
            if use_status:
                status = mpi4py.MPI.Status()

            if use_any_recv:
                y = self.communicator.recv_obj(source=0,
                                               status=status)
            else:
                y = self.communicator.recv_obj(source=0,
                                               tag=tag,
                                               status=status)

            if use_status:
                status_src = status.Get_source()
                self.assertEqual(0, status_src)
                status_tag = status.Get_tag()
                self.assertEqual(tag, status_tag)

        self.assertEqual(x, y)

    def test_send_recv_obj(self):
        self.setup()

        self.check_send_recv_obj(0)
        self.check_send_recv_obj(1, tag=1)
        self.check_send_recv_obj(2, tag=2, use_any_recv=False)

        self.check_send_recv_obj(3, use_status=True)
        self.check_send_recv_obj(4, tag=4, use_status=True)
        self.check_send_recv_obj(5, tag=5, use_any_recv=False, use_status=True)

        self.teardown()

    def test_send_recv_obj_chx_cpu(self):
        self.setup()

        with chainerx.using_device("native"):
            chx_array = chainerx.array([0])
            self.check_send_recv_obj(chx_array)

            chx_array = chainerx.array([1])
            self.check_send_recv_obj(chx_array, tag=1)

            chx_array = chainerx.array([2])
            self.check_send_recv_obj(chx_array, tag=2, use_any_recv=False)

        self.teardown()

    @chainer.testing.attr.gpu
    def test_send_obj_chx_gpu(self):
        self.setup()

        rank_next = (self.communicator.rank + 1) % self.communicator.size
        with chainerx.using_device("cuda"):
            chx_array = chainerx.array([0])
            with pytest.raises(ValueError):
                self.communicator.send_obj(chx_array, dest=rank_next)

            chx_array_list = [[0], chainerx.array([1])]
            with pytest.raises(ValueError):
                self.communicator.send_obj(chx_array_list, dest=rank_next)

            chx_array_tuple = (0, chainerx.array([2]))
            with pytest.raises(ValueError):
                self.communicator.send_obj(chx_array_tuple, dest=rank_next)

            chx_array_dict_value = {0: chainerx.array([2])}
            with pytest.raises(ValueError):
                self.communicator.send_obj(chx_array_dict_value,
                                           dest=rank_next)

            chx_array_dict_key = {chainerx.array([2]): 0}
            with pytest.raises(ValueError):
                self.communicator.send_obj(chx_array_dict_key, dest=rank_next)

            chx_array_dict_set = {chainerx.array([2]), 0}
            with pytest.raises(ValueError):
                self.communicator.send_obj(chx_array_dict_set, dest=rank_next)

        self.teardown()

    @chainer.testing.attr.gpu
    def test_collective_obj_chx_gpu(self):
        self.setup()

        test_function_list = [self.communicator.gather_obj,
                              self.communicator.bcast_obj,
                              self.communicator.allreduce_obj]
        with chainerx.using_device("cuda"):
            for func in test_function_list:
                chx_array = chainerx.array([0])
                with pytest.raises(ValueError):
                    func(chx_array)

                chx_array_list = [[0], chainerx.array([1])]
                with pytest.raises(ValueError):
                    func(chx_array_list)

                chx_array_tuple = (0, chainerx.array([2]))
                with pytest.raises(ValueError):
                    func(chx_array_tuple)

                chx_array_dict_value = {0: chainerx.array([2])}
                with pytest.raises(ValueError):
                    func(chx_array_dict_value)

                chx_array_dict_key = {chainerx.array([2]): 0}
                with pytest.raises(ValueError):
                    func(chx_array_dict_key)

                chx_array_dict_set = {chainerx.array([2]), 0}
                with pytest.raises(ValueError):
                    func(chx_array_dict_set)

        self.teardown()

    def test_config(self):
        self.setup()
        assert self.communicator.batched_copy
        assert self.communicator.get_config('batched_copy')
        self.communicator.set_config('batched_copy', False)
        assert not self.communicator.batched_copy
        assert not self.communicator.get_config('batched_copy')
        self.communicator.set_config('batched_copy')
        assert self.communicator.batched_copy
        assert self.communicator.get_config('batched_copy')

    def test_config_context(self):
        self.setup()

        # Although this is not external interface, but to be tested
        with self.communicator.config_scope():
            self.communicator.foobar = '0xdeadbeef'

        assert '0xdeadbeef' == self.communicator._configs['foobar']
