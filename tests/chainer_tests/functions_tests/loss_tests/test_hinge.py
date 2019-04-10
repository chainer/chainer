import unittest

import numpy
import six

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [{'dtype': numpy.float16,
      'forward_options': {'rtol': 3e-3, 'atol': 3e-3},
      'backward_options': {'rtol': 1e-1, 'atol': 1e-1}},
     {'dtype': numpy.float32,
      'forward_options': {},
      'backward_options': {'rtol': 1e-1, 'atol': 1e-1}},
     {'dtype': numpy.float64,
      'forward_options': {},
      'backward_options': {'rtol': 1e-1, 'atol': 1e-1}},
     ],
    [{'reduce': 'no'},
     {'reduce': 'mean'},
     ],
    [{'norm': 'L1'},
     {'norm': 'L2'},
     ],
    [{'label_dtype': numpy.int8},
     {'label_dtype': numpy.int16},
     {'label_dtype': numpy.int32},
     {'label_dtype': numpy.int64},
     ],
))
class TestHinge(unittest.TestCase):

    def setUp(self):
        self._config_user = chainer.using_config('dtype', self.dtype)
        self._config_user.__enter__()

        shape = (10, 5)
        self.x = numpy.random.uniform(-1, 1, shape).astype(self.dtype)
        # Avoid values around -1.0 for stability
        self.x[numpy.logical_and(-1.01 < self.x, self.x < -0.99)] = 0.5
        self.t = numpy.random.randint(
            0, shape[1], shape[:1]).astype(self.label_dtype)
        if self.reduce == 'no':
            self.gy = numpy.random.uniform(
                -1, 1, self.x.shape).astype(self.dtype)

    def tearDown(self):
        self._config_user.__exit__(None, None, None)

    def check_forward(self, x_data, t_data):
        x_val = chainer.Variable(x_data)
        t_val = chainer.Variable(t_data, requires_grad=False)
        loss = functions.hinge(x_val, t_val, self.norm, self.reduce)
        if self.reduce == 'mean':
            self.assertEqual(loss.data.shape, ())
        else:
            self.assertEqual(loss.data.shape, self.x.shape)
        self.assertEqual(loss.data.dtype, self.dtype)
        loss_value = cuda.to_cpu(loss.data)

        # Compute expected value
        for i in six.moves.range(self.x.shape[0]):
            self.x[i, self.t[i]] *= -1
        for i in six.moves.range(self.x.shape[0]):
            for j in six.moves.range(self.x.shape[1]):
                self.x[i, j] = max(0, 1.0 + self.x[i, j])
        if self.norm == 'L1':
            loss_expect = self.x
        elif self.norm == 'L2':
            loss_expect = self.x ** 2
        if self.reduce == 'mean':
            loss_expect = numpy.sum(loss_expect) / self.x.shape[0]

        testing.assert_allclose(
            loss_expect, loss_value, **self.forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.chainerx
    def test_forward_chainerx_native(self):
        self.check_forward(
            backend.to_chx(self.x), backend.to_chx(self.t))

    @attr.gpu
    @attr.chainerx
    def test_forward_chainerx_cuda(self):
        self.check_forward(
            backend.to_chx(cuda.to_gpu(self.x)),
            backend.to_chx(cuda.to_gpu(self.t)))

    def check_backward(self, x_data, t_data):
        def f(x, t):
            return functions.hinge(x, t, self.norm)

        gradient_check.check_backward(
            f, (x_data, t_data), None, dtype='d', **self.backward_options)

    def check_backward_chainerx(self, x_data, t_data):
        # TODO(niboshi): gradient_check does not support integer input
        # (no_grads) for ChainerX. Support it and merge this method with
        # `self.check_backward`.

        def f(x):
            return functions.hinge(x, t_data, self.norm)

        gradient_check.check_backward(
            f, (x_data,), None, dtype='d', **self.backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.t)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.chainerx
    def test_backward_chainerx_native(self):
        self.check_backward_chainerx(
            backend.to_chx(self.x),
            backend.to_chx(self.t))

    @attr.gpu
    @attr.chainerx
    def test_backward_chainerx_cuda(self):
        self.check_backward_chainerx(
            backend.to_chx(cuda.to_gpu(self.x)),
            backend.to_chx(cuda.to_gpu(self.t)))


class TestHingeInvalidOption(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (10, 5)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 5, (10,)).astype(numpy.int32)

    def check_invalid_norm_option(self, xp):
        x = xp.asarray(self.x)
        t = xp.asarray(self.t)
        with self.assertRaises(NotImplementedError):
            functions.hinge(x, t, 'invalid_norm', 'mean')

    def test_invalid_norm_option_cpu(self):
        self.check_invalid_norm_option(numpy)

    @attr.gpu
    def test_invalid_norm_option_gpu(self):
        self.check_invalid_norm_option(cuda.cupy)

    def check_invalid_reduce_option(self, xp):
        x = xp.asarray(self.x)
        t = xp.asarray(self.t)
        with self.assertRaises(ValueError):
            functions.hinge(x, t, 'L1', 'invalid_option')

    def test_invalid_reduce_option_cpu(self):
        self.check_invalid_reduce_option(numpy)

    @attr.gpu
    def test_invalid_reduce_option_gpu(self):
        self.check_invalid_reduce_option(cuda.cupy)


testing.run_module(__name__, __file__)
