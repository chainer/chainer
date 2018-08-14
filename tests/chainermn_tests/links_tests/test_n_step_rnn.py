import unittest

import chainer
import chainer.cuda
import chainer.functions as F
import chainer.links as L
import chainer.testing
import chainermn
import numpy as np
import pytest


class Model(chainer.Chain):
    def __init__(self, n_vocab, n_hid, communicator, rank_next, rank_prev):
        n_layer = 1
        n_rnn_hid = 10
        super(Model, self).__init__(
            l1=L.EmbedID(n_vocab, n_rnn_hid, ignore_label=-1),
            rnn=chainermn.links.create_multi_node_n_step_rnn(
                L.NStepLSTM(
                    n_layers=n_layer, in_size=n_rnn_hid, out_size=n_rnn_hid,
                    dropout=0.1),
                communicator, rank_in=rank_prev, rank_out=rank_next,
            ),
            l2=L.Linear(n_rnn_hid, n_hid),
            l3=L.Linear(n_hid, 1))

    def __call__(self, xs, ts):
        h1 = [self.l1(x) for x in xs]
        # MultiNodeNStepRNN returns outputs of actual_rnn + delegate_variable.
        cell1, cell2, os, delegate_variable = self.rnn(h1)
        os = F.concat(os, axis=0)
        h2 = self.l2(os)
        h3 = self.l3(h2)
        ys = F.sum(h3, axis=0)
        err = F.mean_squared_error(ys, ts)
        err = chainermn.functions.pseudo_connect(delegate_variable, err)
        return err


class TestNStepRNN(unittest.TestCase):

    def setup(self, gpu):
        if gpu:
            self.communicator = chainermn.create_communicator('hierarchical')
            chainer.cuda.get_device_from_id(self.communicator.intra_rank).use()
        else:
            self.communicator = chainermn.create_communicator('naive')

        if self.communicator.size < 2:
            pytest.skip("This test is for multinode only")

        self.rank_next = self.communicator.rank + 1
        self.rank_prev = self.communicator.rank - 1

        if self.rank_prev < 0:
            self.rank_prev = None

        if self.rank_next >= self.communicator.size:
            self.rank_next = None

    def check_homogeneous_rnn(self, gpu):
        self.setup(gpu)

        n, n_vocab, l = 100, 8, 10
        # Number of model parameters are same among processes.
        n_hid = 2

        X = [np.random.randint(
            0, n_vocab, size=np.random.randint(l // 2, l + 1), dtype=np.int32)
            for _ in range(n)]
        Y = (np.random.rand(n) * 2).astype(np.float32)
        model = Model(
            n_vocab, n_hid, self.communicator, self.rank_next, self.rank_prev)

        if gpu:
            model.to_gpu()
            X = [chainer.cuda.to_gpu(x) for x in X]
            Y = chainer.cuda.to_gpu(Y)

        for i in range(n):
            err = model(X[i:i + 1], Y[i:i + 1])
            err.backward()

        # Check if backprop finishes without deadlock.
        self.assertTrue(True)

    def test_homogeneous_rnn_cpu(self):
        self.check_homogeneous_rnn(False)

    @chainer.testing.attr.gpu
    def test_homogeneous_rnn_gpu(self):
        self.check_homogeneous_rnn(True)

    def check_heterogeneous_rnn(self, gpu):
        self.setup(gpu)

        n, n_vocab, l = 100, 8, 10
        # Number of model parameters are different among processes.
        n_hid = (self.communicator.rank + 1) * 10

        X = [np.random.randint(
            0, n_vocab, size=np.random.randint(l // 2, l + 1), dtype=np.int32)
            for _ in range(n)]
        Y = (np.random.rand(n) * 2).astype(np.float32)
        model = Model(
            n_vocab, n_hid, self.communicator, self.rank_next, self.rank_prev)

        if gpu:
            model.to_gpu()
            X = [chainer.cuda.to_gpu(x) for x in X]
            Y = chainer.cuda.to_gpu(Y)

        for i in range(n):
            err = model(X[i:i + 1], Y[i:i + 1])
            err.backward()

        # Check if backprop finishes without deadlock.
        self.assertTrue(True)

    def test_heterogeneous_rnn_cpu(self):
        self.check_heterogeneous_rnn(False)

    @chainer.testing.attr.gpu
    def test_heterogeneous_rnn_gpu(self):
        self.check_heterogeneous_rnn(True)
