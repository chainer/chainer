import chainer
import chainer.backends
from chainer.backends.cuda import cupy
import chainer.functions as F
import chainer.links as L
import chainer.testing
import chainer.testing.attr
import chainermn
import numpy as np
import pytest


class Model(chainer.Chain):
    def __init__(self, n_vocab, n_hid, communicator, rank_next, rank_prev):
        n_layers = 1
        n_rnn_hid = 10
        super(Model, self).__init__()
        with self.init_scope():
            self.l1 = L.EmbedID(n_vocab, n_rnn_hid, ignore_label=-1)
            self.rnn = chainermn.links.create_multi_node_n_step_rnn(
                L.NStepLSTM(
                    n_layers=n_layers, in_size=n_rnn_hid, out_size=n_rnn_hid,
                    dropout=0.1),
                communicator, rank_in=rank_prev, rank_out=rank_next,
            )
            self.l2 = L.Linear(n_rnn_hid, n_hid)
            self.l3 = L.Linear(n_hid, 1)

    def __call__(self, xs, ts):
        h1 = [self.l1(x) for x in xs]
        # MultiNodeNStepRNN returns outputs of actual_rnn + delegate_variable.
        cell1, cell2, os, delegate_variable = self.rnn(h1)
        os = F.concat(os, axis=0)
        h2 = self.l2(os)
        h3 = self.l3(h2)
        ys = F.sum(h3, axis=0)
        err = F.mean_squared_error(ys, ts)
        err, = chainermn.functions.pseudo_connect(delegate_variable, err)
        return err


def setup_communicator(gpu):
    if gpu:
        communicator = chainermn.create_communicator('flat')
        chainer.backends.cuda.get_device_from_id(
            communicator.intra_rank).use()
    else:
        communicator = chainermn.create_communicator('naive')

    if communicator.size < 2:
        pytest.skip('This test is for multinode only')

    rank_next = communicator.rank + 1
    rank_prev = communicator.rank - 1

    if rank_prev < 0:
        rank_prev = None

    if rank_next >= communicator.size:
        rank_next = None

    return communicator, rank_prev, rank_next


def check_homogeneous_rnn(gpu, dtype):
    communicator, rank_prev, rank_next = setup_communicator(gpu=gpu)

    n, n_vocab, l = 100, 8, 10
    # Number of model parameters are same among processes.
    n_hid = 2
    with chainer.using_config('dtype', dtype):
        X = [np.random.randint(
            0, n_vocab, size=np.random.randint(l // 2, l + 1),
            dtype=np.int32)
            for _ in range(n)]
        Y = (np.random.rand(n) * 2).astype(dtype)
        model = Model(
            n_vocab, n_hid, communicator, rank_next,
            rank_prev)

        if gpu:
            model.to_device(cupy.cuda.Device())
            X = [chainer.backends.cuda.to_gpu(x) for x in X]
            Y = chainer.backends.cuda.to_gpu(Y)

        for i in range(n):
            err = model(X[i:i + 1], Y[i:i + 1])
            err.backward()

        # Check if backprop finishes without deadlock.
        assert True


@pytest.mark.parametrize('dtype', [np.float16, np.float32])
def test_homogeneous_rnn_cpu(dtype):
    check_homogeneous_rnn(False, dtype)


@chainer.testing.attr.gpu
@pytest.mark.parametrize('dtype', [np.float16, np.float32])
def test_homogeneous_rnn_gpu(dtype):
    check_homogeneous_rnn(True, dtype)


def check_heterogeneous_rnn(gpu, dtype):
    communicator, rank_prev, rank_next = setup_communicator(gpu)

    with chainer.using_config('dtype', dtype):
        n, n_vocab, l = 100, 8, 10
        # Number of model parameters are different among processes.
        n_hid = (communicator.rank + 1) * 10

        X = [np.random.randint(
            0, n_vocab, size=np.random.randint(l // 2, l + 1),
            dtype=np.int32)
            for _ in range(n)]
        Y = (np.random.rand(n) * 2).astype(dtype)
        model = Model(
            n_vocab, n_hid, communicator, rank_next,
            rank_prev)

        if gpu:
            model.to_device(cupy.cuda.Device())
            X = [chainer.backends.cuda.to_gpu(x) for x in X]
            Y = chainer.backends.cuda.to_gpu(Y)

        for i in range(n):
            err = model(X[i:i + 1], Y[i:i + 1])
            err.backward()

        # Check if backprop finishes without deadlock.
        assert True


@pytest.mark.parametrize('dtype', [np.float16, np.float32])
def test_heterogeneous_rnn_cpu(dtype):
    check_heterogeneous_rnn(False, dtype)


@chainer.testing.attr.gpu
@pytest.mark.parametrize('dtype', [np.float16, np.float32])
def test_heterogeneous_rnn_gpu(dtype):
    check_heterogeneous_rnn(True, dtype)
