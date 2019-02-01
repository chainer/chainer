import math
import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class CTCTestBase(object):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (4, 2, 3)).astype(self.dtype)
        self.t = numpy.array([[0, 1], [1, 0]]).astype(numpy.int32)
        self.l = numpy.array([[2, 0, 2, 1, 2],
                              [2, 1, 2, 0, 2]]).astype(numpy.int32)
        self.blank_symbol = 2
        self.x_length = numpy.full((len(self.x[0]),), len(self.x), dtype='i')
        self.l_length = numpy.full((len(self.t),), len(self.t[0]), dtype='i')
        self.use_length = True
        if self.reduce == 'mean':
            self.gy = numpy.random.uniform(-1, 1, ()).astype(self.dtype)
        else:
            self.gy = numpy.random.uniform(-1, 1, (2,)).astype(self.dtype)

        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-2}
            self.check_backward_options = {
                'atol': 1e-3, 'dtype': numpy.float64}
        else:
            self.check_forward_options = {}
            self.check_backward_options = {'atol': 1e-4}

    # recursive forward computation.
    def alpha(self, x, l, t, u):
        if u < 0:
            return 0.0
        if t == 0:
            if u == 0:
                return x[0][self.blank_symbol]
            elif u == 1:
                return x[0][l[1]]
            else:
                return 0.0
        elif l[u] == self.blank_symbol or l[u] == l[u - 2]:
            return (x[t][l[u]] *
                    (self.alpha(x, l, t - 1, u - 1) +
                     self.alpha(x, l, t - 1, u)))
        else:
            return (x[t][l[u]] *
                    (self.alpha(x, l, t - 1, u - 2) +
                     self.alpha(x, l, t - 1, u - 1) +
                     self.alpha(x, l, t - 1, u)))

    def check_forward(self, t_data, xs_data, l_length, x_length,
                      wrap_variable=True):
        if wrap_variable:
            x = tuple(chainer.Variable(x_data) for x_data in xs_data)
            t = chainer.Variable(t_data)
        else:
            x = xs_data
            t = t_data

        args = (x, t, self.blank_symbol)
        if self.use_length:
            if wrap_variable:
                args += (chainer.Variable(x_length),
                         chainer.Variable(l_length))
            else:
                args += (x_length, l_length)
        loss = functions.connectionist_temporal_classification(
            *args, reduce=self.reduce).data

        # compute expected value by recursive computation.
        xp = backend.get_array_module(self.x)
        xt = xp.swapaxes(self.x, 0, 1)
        for b in range(xt.shape[0]):
            for t in range(xt.shape[1]):
                xt[b][t] = numpy.exp(xt[b][t]) / numpy.sum(numpy.exp(xt[b][t]))
        batch_size = xt.shape[0]
        path_length = 2 * l_length + 1
        loss_expect = xp.zeros((batch_size,), dtype=self.dtype)
        for i in range(batch_size):
            xtb, lb, xlb, plb = xt[i], self.l[i], x_length[i], path_length[i]
            loss_expect[i] = -math.log(
                self.alpha(xtb, lb, int(xlb - 1), int(plb - 1)) +
                self.alpha(xtb, lb, int(xlb - 1), int(plb - 2)))
        if self.reduce == 'mean':
            loss_expect = xp.mean(loss_expect)
        testing.assert_allclose(
            loss_expect, loss, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.t, tuple(self.x),
                           self.l_length, self.x_length)

    def test_forward_without_wrap_cpu(self):
        self.check_forward(self.t, tuple(self.x),
                           self.l_length, self.x_length, wrap_variable=False)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.t),
                           tuple(cuda.to_gpu(x_data) for x_data in self.x),
                           cuda.to_gpu(self.l_length),
                           cuda.to_gpu(self.x_length))

    @attr.gpu
    def test_forward_without_wrap_gpu(self):
        self.check_forward(cuda.to_gpu(self.t),
                           tuple(cuda.to_gpu(x_data) for x_data in self.x),
                           cuda.to_gpu(self.l_length),
                           cuda.to_gpu(self.x_length),
                           wrap_variable=False)

    # expected value(via numerical differentiation) from t_data
    def check_backward(self, t_data, xs_data, l_length, x_length, gy_data):
        def f(input_length, label_length, t, *x):
            return functions.connectionist_temporal_classification(
                x, t, self.blank_symbol, x_length, l_length,
                reduce=self.reduce)

        gradient_check.check_backward(
            f, (x_length, l_length, t_data) + xs_data, gy_data,
            eps=1e-2, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.t, tuple(self.x),
                            self.l_length, self.x_length,
                            self.gy)

    @condition.retry(3)
    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.t),
                            tuple(cuda.to_gpu(x_data) for x_data in self.x),
                            cuda.to_gpu(self.l_length),
                            cuda.to_gpu(self.x_length),
                            cuda.to_gpu(self.gy))


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'reduce': ['mean', 'no'],
}))
class TestCTC(unittest.TestCase, CTCTestBase):

    def setUp(self):
        CTCTestBase.setUp(self)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'reduce': ['mean', 'no'],
}))
class TestCTCWithoutLength(unittest.TestCase, CTCTestBase):

    def setUp(self):
        CTCTestBase.setUp(self)
        self.use_length = False


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'reduce': ['mean', 'no'],
}))
class TestCTCWithLabelPadding(unittest.TestCase, CTCTestBase):

    def setUp(self):
        CTCTestBase.setUp(self)
        self.l_length[0] = 1


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'reduce': ['mean', 'no'],
}))
class TestCTCWithInputPadding(unittest.TestCase, CTCTestBase):

    def setUp(self):
        CTCTestBase.setUp(self)
        self.x_length[0] = 3


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'reduce': ['mean', 'no'],
}))
class TestCTCWithAllPadding(unittest.TestCase, CTCTestBase):

    def setUp(self):
        CTCTestBase.setUp(self)
        self.x_length[...] = 3
        self.l_length[...] = 1


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'reduce': ['mean', 'no'],
}))
class TestCTCWithRepeatedLabel(unittest.TestCase, CTCTestBase):

    def setUp(self):
        CTCTestBase.setUp(self)
        self.t = numpy.array([[0, 1, 1], [0, 1, 0]]).astype(numpy.int32)
        self.l = numpy.array([[2, 0, 2, 1, 2, 1, 2],
                              [2, 0, 2, 1, 2, 0, 2]]).astype(numpy.int32)
        self.l_length = numpy.full((len(self.t),), len(self.t[0]), dtype='i')


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'reduce': ['mean', 'no'],
}))
class TestCTCBlankSymbol(unittest.TestCase, CTCTestBase):

    def setUp(self):
        CTCTestBase.setUp(self)
        self.x = numpy.random.uniform(-1, 1, (4, 2, 4)).astype(self.dtype)
        self.l = numpy.array([[3, 0, 3, 1, 3],
                              [3, 1, 3, 0, 3]]).astype(numpy.int32)
        self.blank_symbol = 3


class TestCTCUseNoBackpropMode(unittest.TestCase):

    def test_no_backprop_mode(self):
        xs_data = numpy.random.uniform(-1, 1, (4, 2, 3)).astype(numpy.float32)
        t_data = numpy.array([[0, 1], [1, 0]]).astype(numpy.int32)
        with chainer.no_backprop_mode():
            x = [chainer.Variable(x_data) for x_data in xs_data]
            t = chainer.Variable(t_data)
            functions.connectionist_temporal_classification(x, t, 2)


class TestCTCError(unittest.TestCase):

    def test_not_iterable(self):
        x = chainer.Variable(numpy.zeros((4, 2, 3), numpy.float32))
        t = chainer.Variable(numpy.zeros((2, 2), numpy.int32))
        with self.assertRaises(TypeError):
            functions.connectionist_temporal_classification(x, t, 0)


class TestCTCInvalidReductionOption(unittest.TestCase):

    def test_not_iterable(self):
        x = chainer.Variable(numpy.zeros((4, 2, 3), numpy.float32))
        t = chainer.Variable(numpy.zeros((2, 2), numpy.int32))
        with self.assertRaises(ValueError):
            functions.connectionist_temporal_classification(
                tuple(x), t, 0, reduce='invalid_option')


testing.run_module(__name__, __file__)
