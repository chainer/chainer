import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


def _inv(x):
    if x.ndim == 2:
        return numpy.linalg.inv(x)
    return numpy.array([numpy.linalg.inv(ix) for ix in x])


def _make_eye(shape):
    if len(shape) == 2:
        n = shape[0]
        return numpy.eye(n, dtype=numpy.float32)
    m = shape[0]
    n = shape[1]
    return numpy.array([numpy.eye(n, dtype=numpy.float32)] * m)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'shape': [(1, 1), (5, 5)],
}))
@testing.fix_random()
@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1'],
    })
)
class InvFunctionTest(testing.FunctionTestCase):

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_dtype = numpy.float32
            self.check_forward_options.update({'atol': 1e-3, 'rtol': 1e-3})
            self.check_backward_options.update({'atol': 1e-3, 'rtol': 1e-3})
            self.check_double_backward_options.update({
                'atol': 5e-3, 'rtol': 5e-3})
        else:
            self.check_forward_dtype = self.dtype
            self.check_forward_options.update({'atol': 1e-4, 'rtol': 1e-4})
            self.check_backward_options.update({'atol': 5e-4, 'rtol': 5e-4})
            self.check_double_backward_options.update({
                'atol': 5e-4, 'rtol': 5e-4})

    def generate_inputs(self):
        x = (numpy.eye(self.shape[-1]) +
             numpy.random.uniform(-0.01, 0.01, self.shape)).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        return functions.inv(x),

    def forward_expected(self, inputs):
        x, = inputs
        x1 = x.astype(self.check_forward_dtype, copy=False)
        return _inv(x1).astype(self.dtype),

    def test_identity(self, backend_config):
        x, = self.generate_inputs()
        x = chainer.Variable(backend_config.get_array(x))
        y = functions.matmul(x, functions.inv(x))
        testing.assert_allclose(
            y.data, _make_eye(x.shape), **self.check_forward_options)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'shape': [(5, 1, 1), (3, 5, 5)],
}))
@testing.fix_random()
@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1'],
    })
)
class BatchInvFunctionTest(testing.FunctionTestCase):

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_dtype = numpy.float32
            self.check_forward_options.update({'atol': 1e-3, 'rtol': 1e-3})
            self.check_backward_options.update({'atol': 1e-3, 'rtol': 1e-3})
            self.check_double_backward_options.update({
                'atol': 5e-3, 'rtol': 5e-3})
        else:
            self.check_forward_dtype = self.dtype
            self.check_forward_options.update({'atol': 1e-4, 'rtol': 1e-4})
            self.check_backward_options.update({'atol': 5e-4, 'rtol': 5e-4})
            self.check_double_backward_options.update({
                'atol': 1e-3, 'rtol': 1e-3})

    def generate_inputs(self):
        x = (numpy.eye(self.shape[-1]) +
             numpy.random.uniform(-0.01, 0.01, self.shape)).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        return functions.batch_inv(x),

    def forward_expected(self, inputs):
        x, = inputs
        x1 = x.astype(self.check_forward_dtype, copy=False)
        return _inv(x1).astype(self.dtype),

    def test_identity(self, backend_config):
        x, = self.generate_inputs()
        x = chainer.Variable(backend_config.get_array(x))
        y = functions.matmul(x, functions.batch_inv(x))
        testing.assert_allclose(
            y.data, _make_eye(x.shape), **self.check_forward_options)


class InvFunctionRaiseTest(unittest.TestCase):

    def test_invalid_ndim(self):
        x = chainer.Variable(numpy.zeros((1, 2, 2), dtype=numpy.float32))
        with self.assertRaises(type_check.InvalidType):
            functions.inv(x)

    def test_invalid_shape(self):
        x = chainer.Variable(numpy.zeros((1, 2), dtype=numpy.float32))
        with self.assertRaises(type_check.InvalidType):
            functions.inv(x)

    def test_singular_cpu(self):
        x = chainer.Variable(numpy.zeros((2, 2), dtype=numpy.float32))
        with self.assertRaises(ValueError):
            functions.inv(x)

    @attr.gpu
    def test_singular_gpu(self):
        x = chainer.Variable(
            cuda.to_gpu(numpy.zeros((2, 2), dtype=numpy.float32)))

        # Should raise exception only when debug mode.
        with chainer.using_config('debug', False):
            functions.inv(x)

        with chainer.using_config('debug', True):
            with self.assertRaises(ValueError):
                functions.inv(x)


class BatchInvFunctionRaiseTest(unittest.TestCase):

    def test_invalid_ndim(self):
        x = chainer.Variable(numpy.zeros((2, 2), dtype=numpy.float32))
        with self.assertRaises(type_check.InvalidType):
            functions.batch_inv(x)

    def test_invalid_shape(self):
        x = chainer.Variable(numpy.zeros((1, 2, 1), dtype=numpy.float32))
        with self.assertRaises(type_check.InvalidType):
            functions.batch_inv(x)

    def test_singular_cpu(self):
        x = chainer.Variable(numpy.zeros((1, 2, 2), dtype=numpy.float32))
        with self.assertRaises(ValueError):
            functions.batch_inv(x)

    @attr.gpu
    def test_singular_gpu(self):
        x = chainer.Variable(
            cuda.to_gpu(numpy.zeros((1, 2, 2), dtype=numpy.float32)))

        # Should raise exception only when debug mode.
        with chainer.using_config('debug', False):
            functions.batch_inv(x)

        with chainer.using_config('debug', True):
            with self.assertRaises(ValueError):
                functions.batch_inv(x)


testing.run_module(__name__, __file__)
