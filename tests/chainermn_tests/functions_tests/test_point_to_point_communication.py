import copy
import functools

import chainer
import chainer.testing
import chainer.testing.attr
import numpy
import pytest

import chainermn
import chainermn.functions


class Param(object):
    def __init__(self, param):
        self.dtype = None
        self.__dict__.update(param)


class Variables(object):
    def __init__(self, gpu, param):
        self.gpu = gpu
        self.communicator = None
        self.rank_send = 0
        self.rank_recv = 0
        self.f = None
        self.evaluation = None
        self.x = None
        self.model = None
        self.entire_model = None
        self.device = None

        self.setup(param)

    def setup(self, param):
        if self.gpu:
            self.communicator = chainermn.create_communicator('flat')
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

        with chainer.using_config('dtype', param.dtype):
            self.x = chainer.Variable(
                numpy.arange(10).reshape(1, 10).astype(param.dtype) / 10)
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
        return 1.0 * numpy.arange(100).reshape(10, 10) \
            / ((l + 1) * 100)


params = [Param(p) for p in [
    {
        'dtype': numpy.float16,
    }, {
        'dtype': numpy.float32,
    }]]


def check_communication(gpu, param):
    variables = Variables(gpu, param)
    if variables.communicator.rank == 0:
        # Input process.
        y = variables.f(variables.model(variables.x))
        err = chainermn.functions.send(
            y, variables.communicator, variables.rank_send)
        err.backward()
        grad = variables.model.W.grad

        # Compute the expected gradient.
        x_ = variables.x
        for l in range(variables.communicator.size):
            x_ = variables.f(variables.entire_model[l](x_))
        err_ = variables.evaluation(x_, variables.x)
        err_.backward()
        grad_expected = variables.entire_model[0].W.grad

        chainer.testing.assert_allclose(grad, grad_expected)

    elif variables.communicator.rank == variables.communicator.size - 1:
        # Output process.
        x = chainermn.functions.recv(variables.communicator,
                                     variables.rank_recv)
        y = variables.f(variables.model(x))
        err = variables.evaluation(y, variables.x)
        err.backward()

        # Compute the expected output.
        x_ = variables.x
        for l in range(variables.communicator.size):
            x_ = variables.f(variables.entire_model[l](x_))
        y_expect = x_

        chainer.testing.assert_allclose(y.data, y_expect.data)

    else:
        # Intermediate processes.
        x = chainermn.functions.recv(variables.communicator,
                                     variables.rank_recv)
        y = variables.f(variables.model(x))
        err = chainermn.functions.send(
            y, variables.communicator, variables.rank_send)
        err.backward()


@pytest.mark.parametrize('param', params)
def test_communication_cpu(param):
    check_communication(False, param)


@chainer.testing.attr.gpu
@pytest.mark.parametrize('param', params)
def test_communication_gpu(param):
    check_communication(True, param)


def check_retain(gpu, param):
    variables = Variables(gpu, param)

    if variables.communicator.rank == 0:
        # Starting process.
        t = copy.copy(variables.x)
        y = variables.f(variables.model(variables.x))
        dlg = chainermn.functions.send(
            y, variables.communicator, variables.rank_send)

        # Unless delegate_variable is used, backprop would stop here.
        x = chainermn.functions.recv(
            variables.communicator, variables.rank_recv,
            delegate_variable=dlg)
        err = variables.evaluation(x, t)
        err.backward()

        # variables.x.grad is None if backprop stops in the middle.
        assert variables.x.grad is not None

    else:
        # Intermediate processes.
        x = chainermn.functions.recv(variables.communicator,
                                     variables.rank_recv)
        y = variables.f(variables.model(x))
        err = chainermn.functions.send(
            y, variables.communicator, variables.rank_send)
        err.backward()


@pytest.mark.parametrize('param', params)
def test_retain_cpu(param):
    check_retain(False, param)


@chainer.testing.attr.gpu
@pytest.mark.parametrize('param', params)
def test_retain_gpu(param):
    check_retain(True, param)


def check_tuple_communication(length, gpu, param):
    variables = Variables(gpu, param)

    if variables.communicator.rank == 0:
        y = []
        for i in range(length):
            _y = variables.f(variables.model(variables.x))
            y.append(_y)
        err = chainermn.functions.send(
            y, variables.communicator, variables.rank_send)
        err.backward()

    elif variables.communicator.rank == variables.communicator.size - 1:
        y = chainermn.functions.recv(
            variables.communicator, variables.rank_recv, force_tuple=True)
        assert isinstance(y, tuple)
        z = functools.reduce(lambda x, y: x + y, y)
        err = variables.evaluation(z, variables.x)
        err.backward()

    else:
        y = chainermn.functions.recv(variables.communicator,
                                     variables.rank_recv)
        err = chainermn.functions.send(
            y, variables.communicator, variables.rank_send)
        err.backward()


@pytest.mark.parametrize('param', params)
def test_tuple_communication1_cpu(param):
    check_tuple_communication(1, False, param)


@pytest.mark.parametrize('param', params)
def test_tuple_communication2_cpu(param):
    check_tuple_communication(2, False, param)


@chainer.testing.attr.gpu
@pytest.mark.parametrize('param', params)
def test_tuple_communication1_gpu(param):
    check_tuple_communication(1, True, param)


@chainer.testing.attr.gpu
@pytest.mark.parametrize('param', params)
def test_tuple_communication2_gpu(param):
    check_tuple_communication(2, True, param)


# TestNonVariableInput
class NVVariables(object):
    def __init__(self):
        self.communicator = chainermn.create_communicator('naive')

        if self.communicator.size < 2:
            pytest.skip('This test is for multinode')

        self.rank_send = (self.communicator.rank + 1) % self.communicator.size
        self.rank_recv = (self.communicator.rank - 1) % self.communicator.size


@pytest.mark.parametrize('param', params)
def test_non_variable_send(param):
    """Checks if backward will be called even if inputs are not Variable.

    This test confirms whether deadlock occurs when numpy/cupy array is
    given as an input of send.
    In this case, the input will be converted to chainer Variable without
    ``requires_grad``, thus ``backward`` will not be called without any
    modification.
    """
    variables = NVVariables()

    if variables.communicator.rank == 0:
        x = numpy.ones((1, 10)).astype(param.dtype)
        phi = chainermn.functions.send(
            x, variables.communicator, rank=variables.rank_send)
        x, = chainermn.functions.pseudo_connect(phi, x)
        y = chainer.functions.sum(x)
        t = numpy.array(0).astype(param.dtype)
        z = chainer.functions.mean_squared_error(y, t)
        z.backward()

    elif variables.communicator.rank == variables.communicator.size - 1:
        x = chainermn.functions.recv(
            variables.communicator, rank=variables.rank_recv)
        y = chainer.functions.sum(x)
        t = numpy.array(0).astype(param.dtype)
        z = chainer.functions.mean_squared_error(y, t)
        z.backward()

    else:
        x = chainermn.functions.recv(
            variables.communicator, rank=variables.rank_recv)
        phi = chainermn.functions.send(
            x, variables.communicator, rank=variables.rank_send)
        phi.backward()
