import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check
import chainerx


@testing.parameterize(*testing.product_dict(
    [
        {'in_shapes': [(3, 1, 4), (1, 2, 4)], 'out_shape': (3, 2, 4)},
        {'in_shapes': [(3, 2, 4), (4,)], 'out_shape': (3, 2, 4)},
        {'in_shapes': [(3, 2, 4), ()], 'out_shape': (3, 2, 4)},
        {'in_shapes': [(3, 2, 4), (3, 2, 4)], 'out_shape': (3, 2, 4)},
        {'in_shapes': [(), ()], 'out_shape': ()},
        {'in_shapes': [(1, 1, 1), (1,)], 'out_shape': (1, 1, 1)},
        {'in_shapes': [(1, 1, 1), ()], 'out_shape': (1, 1, 1)},
        {'in_shapes': [(3, 2, 4)], 'out_shape': (3, 2, 4)},
        {'in_shapes': [(3, 1, 4), (1, 2, 4), (3, 2, 1)],
         'out_shape': (3, 2, 4)},
        {'in_shapes': [(1, 0, 1), (2,)], 'out_shape': (1, 0, 2)},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
class TestBroadcast(unittest.TestCase):

    def setUp(self):
        uniform = numpy.random.uniform
        self.data = [uniform(0, 1, shape).astype(self.dtype)
                     for shape in self.in_shapes]
        self.grads = [uniform(0, 1, self.out_shape).astype(self.dtype)
                      for _ in range(len(self.in_shapes))]
        self.gg = [uniform(0, 1, shape).astype(self.dtype)
                   for shape in self.in_shapes]

        self.check_backward_options = {}
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_backward_options = {'atol': 1e-2, 'rtol': 1e-1}
            self.check_double_backward_options = {'atol': 1e-2, 'rtol': 1e-1}

    def check_forward(self, data):
        xs = [chainer.Variable(x) for x in data]
        bxs = functions.broadcast(*xs)

        # When len(xs) == 1, function returns a Variable object
        if isinstance(bxs, chainer.Variable):
            bxs = (bxs,)

        for bx in bxs:
            self.assertEqual(bx.data.shape, self.out_shape)
            self.assertEqual(bx.data.dtype, self.dtype)

    def test_forward_cpu(self):
        self.check_forward(self.data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(x) for x in self.data])

    def check_backward(self, data, grads):
        def f(*xs):
            return functions.broadcast(*xs)
        gradient_check.check_backward(
            f, data, grads, dtype=numpy.float64, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.data, self.grads)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward([cuda.to_gpu(x) for x in self.data],
                            [cuda.to_gpu(x) for x in self.grads])

    def check_double_backward(self, data, grads, gg):
        if len(data) == 1:
            return

        gradient_check.check_double_backward(
            functions.broadcast, data, grads, gg, dtype=numpy.float64,
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.data, self.grads, self.gg)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward([cuda.to_gpu(x) for x in self.data],
                                   [cuda.to_gpu(x) for x in self.grads],
                                   [cuda.to_gpu(x) for x in self.gg])


class TestBroadcastTypeError(unittest.TestCase):

    def test_invalid_shape(self):
        x_data = numpy.zeros((3, 2, 5), dtype=numpy.int32)
        y_data = numpy.zeros((1, 3, 4), dtype=numpy.float32)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        with self.assertRaises(type_check.InvalidType):
            functions.broadcast(x, y)

    def test_invalid_shape_fill(self):
        x_data = numpy.zeros((3, 2, 5), dtype=numpy.int32)
        y_data = numpy.zeros(4, dtype=numpy.float32)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        with self.assertRaises(type_check.InvalidType):
            functions.broadcast(x, y)

    def test_no_args(self):
        with self.assertRaises(type_check.InvalidType):
            functions.broadcast()


@testing.parameterize(*testing.product_dict(
    [
        {'in_shape': (3, 1, 4), 'out_shape': (3, 2, 4)},
        {'in_shape': (4,), 'out_shape': (3, 2, 4)},
        {'in_shape': (3, 2, 4), 'out_shape': (3, 2, 4)},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
class TestBroadcastTo(unittest.TestCase):

    def setUp(self):
        uniform = numpy.random.uniform
        self.data = uniform(0, 1, self.in_shape).astype(self.dtype)
        self.grad = uniform(0, 1, self.out_shape).astype(self.dtype)
        self.check_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_backward_options = {
                'eps': 2 ** -5, 'atol': 1e-2, 'rtol': 1e-1}

    def check_forward(self, data):
        x = chainer.Variable(data)
        bx = functions.broadcast_to(x, self.out_shape)

        self.assertEqual(bx.data.shape, self.out_shape)

    def test_forward_cpu(self):
        self.check_forward(self.data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.data))

    @attr.chainerx
    def test_forward_chainerx(self):
        self.check_forward(chainerx.array(self.data))

    def check_backward(self, data, grads):
        gradient_check.check_backward(
            lambda x: functions.broadcast_to(x, self.out_shape), data, grads,
            dtype=numpy.float64, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.data, self.grad)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.data), cuda.to_gpu(self.grad))

    @attr.chainerx
    def test_backward_chainerx(self):
        self.check_backward(
            chainerx.array(self.data), chainerx.array(self.grad))


@testing.parameterize(
    {'in_shape': (3, 2, 4), 'out_shape': (4,)},
    {'in_shape': (3, 2, 4), 'out_shape': (3, 1, 4)},
    {'in_shape': (3, 2, 4), 'out_shape': (1, 3, 2, 3)},
)
class TestBroadcastToTypeCheck(unittest.TestCase):

    def setUp(self):
        uniform = numpy.random.uniform
        self.data = uniform(0, 1, self.in_shape).astype(numpy.float32)

    def test_type_check(self):
        x = chainer.Variable(self.data)
        with self.assertRaises(type_check.InvalidType):
            functions.broadcast_to(x, self.out_shape)


class TestBroadcastToSkip(unittest.TestCase):

    shape = (2, 3)

    def setUp(self):
        self.data = numpy.random.uniform(0, 1, self.shape)

    def test_ndarray(self):
        ret = functions.broadcast_to(self.data, self.shape)
        self.assertIs(self.data, ret.data)

    def test_variable(self):
        x = chainer.Variable(self.data)
        ret = functions.broadcast_to(x, self.shape)
        self.assertIs(x, ret)


testing.run_module(__name__, __file__)
