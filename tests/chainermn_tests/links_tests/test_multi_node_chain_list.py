import chainer
import chainer.cuda
import chainer.links as L
import chainer.testing
import chainermn
import numpy as np
import pytest


class Param(object):
    def __init__(self, param):
        self.dtype = None
        self.__dict__.update(param)


params = [Param(p) for p in [
    {
        'dtype': np.float16,
    }, {
        'dtype': np.float32,
    }]]


class Cycle0SubA(chainer.Chain):
    def __init__(self, size):
        super(Cycle0SubA, self).__init__()
        with self.init_scope():
            self.f = L.Linear(size, size)

    def __call__(self, x):
        return self.f(x)


class Cycle0SubB(chainer.Chain):
    def __init__(self, size):
        super(Cycle0SubB, self).__init__(
            f=L.Linear(size, 2))

    def __call__(self, h):
        return self.f(h)


class Cycle0(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_prev, rank_next):
        super(Cycle0, self).__init__(comm=comm)
        self.add_link(Cycle0SubA(size), rank_in=None, rank_out=rank_next)
        self.add_link(Cycle0SubB(size), rank_in=rank_prev, rank_out=None)


class Cycle1Sub(chainer.Chain):
    def __init__(self, size):
        super(Cycle1Sub, self).__init__(
            f=L.Linear(size, size))

    def __call__(self, h):
        return self.f(h)


class Cycle1(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_prev, rank_next):
        super(Cycle1, self).__init__(comm=comm)
        self.add_link(Cycle1Sub(size), rank_in=rank_prev, rank_out=rank_next)


class Cross0(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_prev, rank_next):
        super(Cross0, self).__init__(comm=comm)
        self.add_link(Cycle0SubA(size), rank_in=None, rank_out=rank_next)
        self.add_link(Cycle0SubB(size), rank_in=rank_prev, rank_out=None)


class Cross1(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_prev, rank_next):
        super(Cross1, self).__init__(comm=comm)
        self.add_link(Cycle0SubB(size), rank_in=rank_prev, rank_out=None)
        self.add_link(Cycle0SubA(size), rank_in=None, rank_out=rank_next)


class BranchSubA(chainer.Chain):
    def __init__(self, size):
        super(BranchSubA, self).__init__(
            f=L.Linear(size, size))

    def __call__(self, x):
        return self.f(x)


class BranchSubB(chainer.Chain):
    def __init__(self, size):
        super(BranchSubB, self).__init__(
            f=L.Linear(size, size))

    def __call__(self, *xs):
        x = xs[0]
        for _x in xs[1:]:
            x = x + _x
        return self.f(x)


class BranchParent1(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_children):
        super(BranchParent1, self).__init__(comm=comm)
        self.add_link(BranchSubA(size), rank_in=None, rank_out=rank_children)
        self.add_link(BranchSubB(size), rank_in=rank_children, rank_out=None)


class BranchParent2(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_children):
        super(BranchParent2, self).__init__(comm=comm)
        ranks = [comm.rank] + rank_children
        self.add_link(BranchSubA(size), rank_in=None, rank_out=ranks)
        self.add_link(BranchSubA(size), rank_in=comm.rank, rank_out=comm.rank)
        self.add_link(BranchSubB(size), rank_in=ranks, rank_out=None)


class BranchParent3(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_children):
        super(BranchParent3, self).__init__(comm=comm)
        ranks = rank_children + [comm.rank]
        self.add_link(BranchSubA(size), rank_in=None, rank_out=ranks)
        self.add_link(BranchSubA(size), rank_in=comm.rank, rank_out=comm.rank)
        self.add_link(BranchSubB(size), rank_in=ranks, rank_out=None)


class BranchParent4(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_children):
        super(BranchParent4, self).__init__(comm=comm)
        ranks = rank_children + [comm.rank]
        ranks = ranks[1:] + ranks[0:1]
        self.add_link(BranchSubA(size), rank_in=None, rank_out=ranks)
        self.add_link(BranchSubA(size), rank_in=comm.rank, rank_out=comm.rank)
        self.add_link(BranchSubB(size), rank_in=ranks, rank_out=None)


class BranchChild(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_parent):
        super(BranchChild, self).__init__(comm=comm)
        self.add_link(
            BranchSubA(size),
            rank_in=rank_parent,
            rank_out=rank_parent)


class TwistFirst(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_next):
        super(TwistFirst, self).__init__(comm=comm)
        self.add_link(BranchSubA(size), rank_in=None, rank_out=rank_next)
        self.add_link(BranchSubA(size), rank_in=rank_next, rank_out=None)


class Twist(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_prev, rank_next):
        super(Twist, self).__init__(comm=comm)
        self.add_link(BranchSubA(size), rank_in=rank_prev, rank_out=comm.rank)
        self.add_link(BranchSubA(size), rank_in=None, rank_out=rank_prev)
        self.add_link(BranchSubA(size), rank_in=None, rank_out=rank_next)
        self.add_link(BranchSubA(size), rank_in=rank_next, rank_out=comm.rank)
        self.add_link(
            BranchSubB(size),
            rank_in=[comm.rank, comm.rank],
            rank_out=None)


class TwistLast(chainermn.MultiNodeChainList):
    def __init__(self, size, comm, rank_prev):
        super(TwistLast, self).__init__(comm=comm)
        self.add_link(BranchSubA(size), rank_in=rank_prev, rank_out=None)
        self.add_link(BranchSubA(size), rank_in=None, rank_out=rank_prev)


class TupleDataSubA(chainer.Chain):
    def __init__(self, size):
        super(TupleDataSubA, self).__init__(
            f0=L.Linear(size, size),
            f1=L.Linear(size, size))

    def __call__(self, x):
        y0 = self.f0(x)
        y1 = self.f1(x)
        return y0, y1


class TupleDataSubB(chainer.Chain):
    def __init__(self, size):
        super(TupleDataSubB, self).__init__(
            f0=L.Linear(size, size),
            f1=L.Linear(size, size))

    def __call__(self, x):
        # TupleDataSubB receives two elemental tuple from TupleDataSubA.
        x0, x1 = x
        y0 = self.f0(x0)
        y1 = self.f1(x1)
        return y0 + y1


class TupleDataSubC(chainer.Chain):
    def __init__(self, size):
        super(TupleDataSubC, self).__init__(
            f=L.Linear(size, size))

    def __call__(self, x):
        return self.f(x)


class TupleDataParent(chainermn.MultiNodeChainList):
    def __init__(self, comm, size, rank_child):
        super(TupleDataParent, self).__init__(comm=comm)
        self.add_link(TupleDataSubA(size), rank_in=None, rank_out=rank_child)
        self.add_link(TupleDataSubC(size), rank_in=rank_child, rank_out=None)


class TupleDataChild(chainermn.MultiNodeChainList):
    def __init__(self, comm, size, rank_parent):
        super(TupleDataChild, self).__init__(comm=comm)
        self.add_link(
            TupleDataSubB(size), rank_in=rank_parent, rank_out=rank_parent)


def create_communicator(gpu):
    if gpu:
        communicator = chainermn.create_communicator('flat')
        chainer.cuda.get_device_from_id(communicator.intra_rank).use()
    else:
        communicator = chainermn.create_communicator('naive')

    if communicator.size < 2:
        pytest.skip('This test is for multinode only')

    rank_next = (communicator.rank + 1) % communicator.size
    rank_prev = (communicator.rank - 1) % communicator.size
    return communicator, rank_next, rank_prev


def check_cycle_model(gpu, param):
    communicator, rank_next, rank_prev = create_communicator(gpu)

    n, d = 100, 10

    with chainer.using_config('dtype', param.dtype):
        if communicator.rank == 0:
            X = np.random.randn(n, d).astype(param.dtype)
            Y = (np.random.rand(n) * 2).astype(np.int32)
            model = L.Classifier(
                Cycle0(d, communicator, rank_next, rank_prev))

            if gpu:
                model.to_gpu()
                X = chainer.cuda.to_gpu(X)
                Y = chainer.cuda.to_gpu(Y)

            for i in range(n):
                err = model(X[i:i + 1], Y[i:i + 1])
                err.backward()
        else:
            model = Cycle1(
                d, communicator, rank_next, rank_prev)
            if gpu:
                model.to_gpu()

            for i in range(n):
                err = model()
                err.backward()


@pytest.mark.parametrize('param', params)
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_cycle_model_cpu(param):
    check_cycle_model(False, param)


@pytest.mark.parametrize('param', params)
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@chainer.testing.attr.gpu
def test_cycle_model_gpu(param):
    check_cycle_model(True, param)


def check_crossing_model(gpu, param):
    communicator, rank_next, rank_prev = create_communicator(gpu)

    n, d = 100, 10
    X = np.random.randn(n, d).astype(param.dtype)
    Y = (np.random.rand(n) * 2).astype(np.int32)

    with chainer.using_config('dtype', param.dtype):
        if communicator.rank == 0:
            model = L.Classifier(Cross0(
                d, communicator, rank_next, rank_prev))
        else:
            model = L.Classifier(Cross1(
                d, communicator, rank_next, rank_prev))

        if gpu:
            model.to_gpu()
            X = chainer.cuda.to_gpu(X)
            Y = chainer.cuda.to_gpu(Y)

        for i in range(n):
            err = model(X[i:i + 1], Y[i:i + 1])
            err.backward()


@pytest.mark.parametrize('param', params)
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_crossing_model_cpu(param):
    check_crossing_model(False, param)


@pytest.mark.parametrize('param', params)
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@chainer.testing.attr.gpu
def test_crossing_model_gpu(param):
    check_crossing_model(True, param)


def check_branching_model(gpu, communicator, rank_next, rank_prev,
                          parent_model, param):
    n, d = 100, 10
    X = np.random.randn(n, d).astype(param.dtype)
    Y = (np.random.rand(n) * 2).astype(np.int32)

    with chainer.using_config('dtype', param.dtype):
        if communicator.rank == 0:
            rank_children = [rank for rank in range(1, communicator.size)]
            model = L.Classifier(parent_model(
                d, communicator, rank_children))
            if gpu:
                model.to_gpu()
                X = chainer.cuda.to_gpu(X)
                Y = chainer.cuda.to_gpu(Y)

            for i in range(n):
                err = model(X[i:i + 1], Y[i:i + 1])
                err.backward()
        else:
            model = BranchChild(d, communicator, 0)
            if gpu:
                model.to_gpu()

            for i in range(n):
                err = model()
                err.backward()


def check_branching_models(gpu, param):
    communicator, rank_next, rank_prev = create_communicator(gpu)
    check_branching_model(gpu, communicator, rank_next, rank_prev,
                          BranchParent1, param)

    check_branching_model(gpu, communicator, rank_next, rank_prev,
                          BranchParent2, param)

    check_branching_model(gpu, communicator, rank_next, rank_prev,
                          BranchParent3, param)

    check_branching_model(gpu, communicator, rank_next, rank_prev,
                          BranchParent4, param)


@pytest.mark.parametrize('param', params)
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_branching_models_cpu(param):
    check_branching_models(False, param)


@pytest.mark.parametrize('param', params)
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@chainer.testing.attr.gpu
def test_branching_models_gpu(param):
    check_branching_models(True, param)


def check_twisting_model(gpu, param):
    communicator, rank_next, rank_prev = create_communicator(gpu)

    n, d = 100, 10
    X = np.random.randn(n, d).astype(param.dtype)
    Y = (np.random.rand(n) * 2).astype(np.int32)

    with chainer.using_config('dtype', param.dtype):
        if communicator.rank == 0:
            model = L.Classifier(
                TwistFirst(d, communicator, rank_next))
        elif communicator.rank == communicator.size - 1:
            model = L.Classifier(
                TwistLast(d, communicator, rank_prev))
        else:
            model = L.Classifier(Twist(
                d, communicator, rank_prev, rank_next))

        if gpu:
            model.to_gpu()
            X = chainer.cuda.to_gpu(X)
            Y = chainer.cuda.to_gpu(Y)

        for i in range(n):
            err = model(X[i:i + 1], Y[i:i + 1])
            err.backward()


@pytest.mark.parametrize('param', params)
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_twisting_model_cpu(param):
    check_twisting_model(False, param)


@pytest.mark.parametrize('param', params)
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@chainer.testing.attr.gpu
def test_twisting_model_gpu(param):
    check_twisting_model(True, param)


def check_tuple_data_model(gpu, param):
    # This test only uses pairs (0, 1), (2, 3), ... (2m, 2m+1)
    communicator, rank_next, rank_prev = create_communicator(gpu)

    n, d = 100, 10
    X = np.random.randn(n, d).astype(param.dtype)
    Y = (np.random.rand(n) * 2).astype(np.int32)

    with chainer.using_config('dtype', param.dtype):
        if communicator.rank % 2 == 0:
            if communicator.rank == communicator.size - 1:
                # in case 2m is the right end with odd number of nodes
                return
            model = L.Classifier(
                TupleDataParent(communicator, d, rank_next))
        elif communicator.rank % 2 == 1:
            model = TupleDataChild(communicator, d, rank_prev)

        assert model is not None
        if gpu:
            model.to_gpu()
            X = chainer.cuda.to_gpu(X)
            Y = chainer.cuda.to_gpu(Y)

        for i in range(n):
            if communicator.rank % 2 == 0:
                err = model(X[i:i + 1], Y[i:i + 1])
            elif communicator.rank % 2 == 1:
                err = model()
            assert err is not None
            err.backward()


@pytest.mark.parametrize('param', params)
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
def test_tuple_data_model_cpu(param):
    check_tuple_data_model(False, param)


@pytest.mark.parametrize('param', params)
@pytest.mark.filterwarnings('ignore::DeprecationWarning')
@chainer.testing.attr.gpu
def test_tuple_data_model_gpu(param):
    check_tuple_data_model(True, param)
