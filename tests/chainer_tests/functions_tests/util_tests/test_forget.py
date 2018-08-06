import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer import variable


@testing.parameterize(*testing.product({
    'out_len': [1, 2],
}))
class TestForget(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.y = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.gz0 = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.gz1 = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.ggx = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.ggy = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)

        self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}
        self.check_double_backward_options = {'atol': 5e-3, 'rtol': 5e-2}

    def check_forward(self, x_data, y_data):
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)
        if self.out_len == 1:
            z = functions.forget(lambda x, y: (x + y + x,), x, y)
            testing.assert_allclose(x_data + y_data + x_data, z.data)
        elif self.out_len == 2:
            z = functions.forget(lambda x, y: (x + y + x, x * y), x, y)
            testing.assert_allclose(x_data + y_data + x_data, z[0].data)
            testing.assert_allclose(x_data * y_data, z[1].data)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.y)

    def check_backward(self, x_data, y_data, *gz_data):
        def f(x, y):
            if self.out_len == 1:
                return functions.forget(lambda x, y: (x + y + x), x, y)
            elif self.out_len == 2:
                return functions.forget(lambda x, y: (x + y + x, x * y), x, y)

        gradient_check.check_backward(
            f, (x_data, y_data), gz_data, **self.check_backward_options)

    def test_backward_cpu(self):
        if self.out_len == 1:
            self.check_backward(self.x, self.y, self.gz0)
        elif self.out_len == 2:
            self.check_backward(self.x, self.y, self.gz0, self.gz1)

    @attr.gpu
    def test_backward_gpu(self):
        if self.out_len == 1:
            self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.y),
                                cuda.to_gpu(self.gz0))
        elif self.out_len == 2:
            self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.y),
                                cuda.to_gpu(self.gz0), cuda.to_gpu(self.gz1))


class TestForgetError(unittest.TestCase):

    def setUp(self):
        self.v = chainer.Variable(numpy.zeros(1))

    def test_not_callable(self):
        with self.assertRaises(TypeError):
            functions.forget(1)

    def test_invalid_type(self):
        with six.assertRaisesRegex(self, RuntimeError, 'int'):
            functions.forget(lambda: 1)

    def test_invalid_tuple_type_1st(self):
        with six.assertRaisesRegex(self, RuntimeError, '1st.*int'):
            functions.forget(lambda: (1,))

    def test_invalid_tuple_type_2nd(self):
        with six.assertRaisesRegex(self, RuntimeError, '2nd.*int'):
            functions.forget(lambda: (self.v, 1))

    def test_invalid_tuple_type_3rd(self):
        with six.assertRaisesRegex(self, RuntimeError, '3rd.*int'):
            functions.forget(lambda: (self.v, self.v, 1))

    def test_invalid_tuple_type_4th(self):
        with six.assertRaisesRegex(self, RuntimeError, '4th.*int'):
            functions.forget(lambda: (self.v,) * 3 + (1,))

    def test_invalid_tuple_type_11th(self):
        with six.assertRaisesRegex(self, RuntimeError, '11th.*int'):
            functions.forget(lambda: (self.v,) * 10 + (1,))

    def test_invalid_tuple_type_12th(self):
        with six.assertRaisesRegex(self, RuntimeError, '12th.*int'):
            functions.forget(lambda: (self.v,) * 11 + (1,))

    def test_invalid_tuple_type_13th(self):
        with six.assertRaisesRegex(self, RuntimeError, '13th.*int'):
            functions.forget(lambda: (self.v,) * 12 + (1,))


class TestForgetGrad(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.x = variable.Variable(self.x)
        self.w = numpy.random.uniform(-1, 1, (3, 2)).astype(numpy.float32)
        self.w = variable.Variable(self.w)
        self.func = lambda a, b: a + b

    def test_grad(self):
        y = functions.forget(self.func, self.x, self.w)
        y.grad_var = variable.Variable(numpy.ones_like(y.data))
        y.backward()

        assert self.x.grad_var is not None
        assert self.w.grad_var is not None


testing.run_module(__name__, __file__)
