import unittest

import numpy

from chainer.backends import cuda
from chainer import gradient_check, testing, Variable, no_backprop_mode
from chainer.functions import hinge_max_margin
from chainer.testing import attr

asl = 'along_second_axis'


@testing.parameterize(*testing.product({
    'reduce': [asl, 'mean'],
    'norm': ['L1', 'L2', 'Huber'],
    'label_dtype': [numpy.int8, numpy.int16, numpy.int32, numpy.int64],
}))
class TestHingeMaxMargin(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(0)
        shape = (200, 3)
        self.x = numpy.random.uniform(-1, 1, shape).astype(numpy.float32)
        self.t = numpy.random.randint(
            0, shape[1], shape[:1]).astype(self.label_dtype)
        self.x[numpy.arange(shape[0]), self.t] += 1
        if self.reduce == asl:
            self.gy = numpy.random.uniform(
                -1, 1, self.x.shape).astype(numpy.float32)

        self.check_backward_options = {'atol': 1e-3, 'rtol': 1e-3}

    def check_forward(self, x_data, t_data):
        x_val = Variable(x_data)
        t_val = Variable(t_data)

        loss = hinge_max_margin(x_val, t_val, self.norm, self.reduce)
        if self.reduce == 'mean':
            self.assertEqual(loss.data.shape, ())
        else:
            new_shape = list(x_data.shape)  # list
            del new_shape[1]
            self.assertEqual(list(loss.data.shape), new_shape)
        self.assertEqual(loss.data.dtype, numpy.float32)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    def check_backward(self, x_data, t_data):
        xp = cuda.get_array_module(x_data)
        if self.norm == 'L1':  # L2 and Huber are differentiable
            no_sign_change = False
            delta = self.check_backward_options['atol'] * 5
            t_data_numpy = cuda.to_cpu(t_data)
            while not no_sign_change:
                samples = len(x_data)
                x_data_plus = cuda.to_cpu(x_data.copy())
                x_data_plus[numpy.arange(samples), t_data_numpy] += delta
                x_data_minus = cuda.to_cpu(x_data.copy())
                x_data_minus[numpy.arange(samples), t_data_numpy] -= delta
                with no_backprop_mode():
                    no_sign_change = numpy.allclose(numpy.sign(
                        hinge_max_margin(x_data_plus, t_data_numpy, self.norm,
                                         asl).data),
                        numpy.sign(hinge_max_margin(x_data_minus, t_data_numpy,
                                                    self.norm, asl).data))
                if not no_sign_change:
                    x_data = xp.random.uniform(-1, 1, x_data.shape).astype(
                        x_data.dtype)
                    x_data[numpy.arange(x_data.shape[0]), t_data] += 1
        else:
            def f(x, t):
                y = hinge_max_margin(x, t, norm=self.norm)
                return y
            gradient_check.check_backward(
                f, (x_data, t_data), None,
                dtype='d',
                **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.t)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))


class TestHingeMaxMarginInvalidOption(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (10, 5)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 5, (10,)).astype(numpy.int32)

    def check_invalid_norm_option(self, xp):
        x = xp.asarray(self.x)
        t = xp.asarray(self.t)
        with self.assertRaises(NotImplementedError):
            hinge_max_margin(x, t, 'invalid_norm', 'mean')

    def test_invalid_norm_option_cpu(self):
        self.check_invalid_norm_option(numpy)

    @attr.gpu
    def test_invalid_norm_option_gpu(self):
        self.check_invalid_norm_option(cuda.cupy)

    def check_invalid_reduce_option(self, xp):
        x = xp.asarray(self.x)
        t = xp.asarray(self.t)
        with self.assertRaises(ValueError):
            hinge_max_margin(x, t, 'L1', 'invalid_option')

    def test_invalid_reduce_option_cpu(self):
        self.check_invalid_reduce_option(numpy)

    @attr.gpu
    def test_invalid_reduce_option_gpu(self):
        self.check_invalid_reduce_option(cuda.cupy)


testing.run_module(__name__, __file__)
