import copy
import functools
import unittest

import chainer
import chainer.testing
import chainer.testing.attr
import numpy
import pytest

import chainermn
import chainermn.functions


class TestPointToPointCommunication(unittest.TestCase):

    def setup(self, gpu):
        self.gpu = gpu
        if self.gpu:
            self.communicator = chainermn.create_communicator('hierarchical')
            device = self.communicator.intra_rank
            chainer.cuda.get_device_from_id(device).use()
        else:
            self.communicator = chainermn.create_communicator('naive')
            device = -1

        if self.communicator.size < 2:
            pytest.skip('This test is for multinode')

        self.rank_send = (self.communicator.rank + 1) % self.communicator.size
        self.rank_recv = (self.communicator.rank - 1) % self.communicator.size

        # Activation function.
        self.f = chainer.functions.sigmoid

        # Evaluation function.
        self.evaluation = chainer.functions.mean_squared_error

        # Input data.
        self.x = chainer.Variable(
            numpy.arange(10).reshape(1, 10).astype(numpy.float32) / 10)

        self.model = chainer.links.Linear(
            10, 10, initialW=self._init_w(self.communicator.rank))
        self.entire_model = [chainer.links.Linear(
            10, 10, initialW=self._init_w(l))
            for l in range(self.communicator.size)]
        self.device = device

        if device >= 0:
            self.x.to_gpu()
            self.model.to_gpu()
            for model in self.entire_model:
                model.to_gpu()

    def _init_w(self, l):
        return 1.0 * numpy.arange(100).reshape(10, 10).astype(numpy.float32) \
            / ((l + 1) * 100)

    def check_communication(self):
        if self.communicator.rank == 0:
            # Input process.
            y = self.f(self.model(self.x))
            err = chainermn.functions.send(
                y, self.communicator, self.rank_send)
            err.backward()
            grad = self.model.W.grad

            # Compute the expected gradient.
            x_ = self.x
            for l in range(self.communicator.size):
                x_ = self.f(self.entire_model[l](x_))
            err_ = self.evaluation(x_, self.x)
            err_.backward()
            grad_expected = self.entire_model[0].W.grad

            chainer.testing.assert_allclose(grad, grad_expected)

        elif self.communicator.rank == self.communicator.size - 1:
            # Output process.
            x = chainermn.functions.recv(self.communicator, self.rank_recv)
            y = self.f(self.model(x))
            err = self.evaluation(y, self.x)
            err.backward()

            # Compute the expected output.
            x_ = self.x
            for l in range(self.communicator.size):
                x_ = self.f(self.entire_model[l](x_))
            y_expect = x_

            chainer.testing.assert_allclose(y.data, y_expect.data)

        else:
            # Intermediate processes.
            x = chainermn.functions.recv(self.communicator, self.rank_recv)
            y = self.f(self.model(x))
            err = chainermn.functions.send(
                y, self.communicator, self.rank_send)
            err.backward()

    def test_communication_cpu(self):
        self.setup(False)
        self.check_communication()

    @chainer.testing.attr.gpu
    def test_communication_gpu(self):
        self.setup(True)
        self.check_communication()

    def check_retain(self):
        if self.communicator.rank == 0:
            # Starting process.
            t = copy.copy(self.x)
            y = self.f(self.model(self.x))
            dlg = chainermn.functions.send(
                y, self.communicator, self.rank_send)

            # Unless delegate_variable is used, backprop would stop here.
            x = chainermn.functions.recv(
                self.communicator, self.rank_recv,
                delegate_variable=dlg)
            err = self.evaluation(x, t)
            err.backward()

            # self.x.grad is None if backprop stops in the middle.
            assert self.x.grad is not None

        else:
            # Intermediate processes.
            x = chainermn.functions.recv(self.communicator, self.rank_recv)
            y = self.f(self.model(x))
            err = chainermn.functions.send(
                y, self.communicator, self.rank_send)
            err.backward()

    def test_retain_cpu(self):
        self.setup(False)
        self.check_retain()

    @chainer.testing.attr.gpu
    def test_retain_gpu(self):
        self.setup(True)
        self.check_retain()

    def check_tuple_communication(self, length):
        if self.communicator.rank == 0:
            y = []
            for i in range(length):
                _y = self.f(self.model(self.x))
                y.append(_y)
            err = chainermn.functions.send(
                y, self.communicator, self.rank_send)
            err.backward()

        elif self.communicator.rank == self.communicator.size - 1:
            y = chainermn.functions.recv(
                self.communicator, self.rank_recv, force_tuple=True)
            assert isinstance(y, tuple)
            z = functools.reduce(lambda x, y: x + y, y)
            err = self.evaluation(z, self.x)
            err.backward()

        else:
            y = chainermn.functions.recv(self.communicator, self.rank_recv)
            err = chainermn.functions.send(
                y, self.communicator, self.rank_send)
            err.backward()

    def test_tuple_communication1_cpu(self):
        self.setup(False)
        self.check_tuple_communication(1)

    def test_tuple_communication2_cpu(self):
        self.setup(False)
        self.check_tuple_communication(2)

    @chainer.testing.attr.gpu
    def test_tuple_communication1_gpu(self):
        self.setup(True)
        self.check_tuple_communication(1)

    @chainer.testing.attr.gpu
    def test_tuple_communication2_gpu(self):
        self.setup(True)
        self.check_tuple_communication(2)


class TestNonVariableInput(unittest.TestCase):

    def setUp(self):
        self.communicator = chainermn.create_communicator('naive')

        if self.communicator.size < 2:
            pytest.skip('This test is for multinode')

        self.rank_send = (self.communicator.rank + 1) % self.communicator.size
        self.rank_recv = (self.communicator.rank - 1) % self.communicator.size

    def test_non_variable_send(self):
        """Checks if backward will be called even if inputs are not Variable.

        This test confirms whether deadlock occurs when numpy/cupy array is
        given as an input of send.
        In this case, the input will be converted to chainer Variable without
        ``requires_grad``, thus ``backward`` will not be called without any
        modification.
        """
        if self.communicator.rank == 0:
            x = numpy.ones((1, 10)).astype(numpy.float32)
            phi = chainermn.functions.send(
                x, self.communicator, rank=self.rank_send)
            x, = chainermn.functions.pseudo_connect(phi, x)
            y = chainer.functions.sum(x)
            t = numpy.array(0).astype(numpy.float32)
            z = chainer.functions.mean_squared_error(y, t)
            z.backward()

        elif self.communicator.rank == self.communicator.size - 1:
            x = chainermn.functions.recv(
                self.communicator, rank=self.rank_recv)
            y = chainer.functions.sum(x)
            t = numpy.array(0).astype(numpy.float32)
            z = chainer.functions.mean_squared_error(y, t)
            z.backward()

        else:
            x = chainermn.functions.recv(
                self.communicator, rank=self.rank_recv)
            phi = chainermn.functions.send(
                x, self.communicator, rank=self.rank_send)
            phi.backward()
