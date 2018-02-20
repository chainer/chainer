import unittest

import mock
import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import backend


def _to_fcontiguous(arrays):
    xp = cuda.get_array_module(*arrays)
    return [xp.asfortranarray(a) for a in arrays]


@testing.parameterize(*testing.product({
    'cover_all': [True, False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'c_contiguous': [True, False],
}))
@backend.inject_backend_tests(
    ['test_forward',
     'test_forward_output_size_zero',
     'test_backward',
     'test_double_backward'],
    # CPU tests
    testing.product({
        'use_cuda': [False],
        'use_ideep': ['never', 'always'],
    })
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
    }))
class TestMaxPooling2D(unittest.TestCase):

    def setUp(self):
        dtype = self.dtype

        # Avoid unstability of numerical gradient
        x = numpy.arange(2 * 3 * 4 * 3, dtype=dtype).reshape(2, 3, 4, 3)
        numpy.random.shuffle(x)
        x = 2 * x / x.size - 1
        if self.cover_all:
            gy = numpy.random.uniform(-1, 1, (2, 3, 3, 2)).astype(dtype)
        else:
            gy = numpy.random.uniform(-1, 1, (2, 3, 2, 2)).astype(dtype)
        ggx = numpy.random.uniform(-1, 1, (2, 3, 4, 3)).astype(dtype)

        self.output_shape = gy.shape

        self.inputs = [x]
        self.grad_outputs = [gy]
        self.grad_grad_inputs = [ggx]

        if self.dtype == numpy.float16:
            self.check_backward_options = {
                'atol': 1e-3, 'rtol': 1e-2}
            self.check_double_backward_options = {
                'atol': 1e-3, 'rtol': 1e-2}
        else:
            self.check_backward_options = {
                'atol': 1e-4, 'rtol': 1e-3}
            self.check_double_backward_options = {
                'atol': 1e-4, 'rtol': 1e-3}

    def forward_cpu(self, inputs):
        x, = inputs
        expect = numpy.empty(self.output_shape, dtype=self.dtype)
        for k in six.moves.range(2):
            for c in six.moves.range(3):
                xx = x[k, c]
                if self.cover_all:
                    expect[k, c] = numpy.array([
                        [xx[0:2, 0:2].max(), xx[0:2, 1:3].max()],
                        [xx[1:4, 0:2].max(), xx[1:4, 1:3].max()],
                        [xx[3:4, 0:2].max(), xx[3:4, 1:3].max()]])
                else:
                    expect[k, c] = numpy.array([
                        [xx[0:2, 0:2].max(), xx[0:2, 1:3].max()],
                        [xx[1:4, 0:2].max(), xx[1:4, 1:3].max()]])
        return expect,

    def check_forward(self, inputs, backend_config):
        y_expect, = self.forward_cpu(inputs)

        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
        if not self.c_contiguous:
            inputs = _to_fcontiguous(inputs)

        with backend_config:
            x, = inputs
            y = functions.max_pooling_2d(x, 3, stride=2, pad=1,
                                         cover_all=self.cover_all)
        assert y.data.dtype == self.dtype
        y_data = cuda.to_cpu(y.data)

        assert self.output_shape == y_data.shape
        testing.assert_allclose(y_expect, y_data)

    def test_forward(self, backend_config):
        self.check_forward(self.inputs, backend_config)

    def test_forward_cpu_wide(self):  # see #120
        x_data = numpy.random.rand(2, 3, 15, 15).astype(self.dtype)
        x = chainer.Variable(x_data)
        functions.max_pooling_2d(x, 6, stride=6, pad=0)

    def test_forward_output_size_zero(self, backend_config):
        with six.assertRaisesRegex(
                self, AssertionError,
                'Height in the output should be positive.'):
            x = numpy.random.rand(4, 4, 1, 4).astype(self.dtype)
            if backend_config.use_cuda:
                x = cuda.to_gpu(x)
            x = chainer.Variable(x)
            with backend_config:
                functions.max_pooling_2d(x, 3, stride=2)

        with six.assertRaisesRegex(
                self, AssertionError,
                'Width in the output should be positive.'):
            x = numpy.random.rand(4, 4, 4, 1).astype(self.dtype)
            if backend_config.use_cuda:
                x = cuda.to_gpu(x)
            x = chainer.Variable(x)
            with backend_config:
                functions.max_pooling_2d(x, 3, stride=2)

    def check_backward(self, inputs, grad_outputs, backend_config):
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)
        if not self.c_contiguous:
            inputs = _to_fcontiguous(inputs)
            grad_outputs = _to_fcontiguous(grad_outputs)

        def f(x):
            return functions.max_pooling_2d(
                x, 3, stride=2, pad=1, cover_all=self.cover_all)

        with backend_config:
            gradient_check.check_backward(
                f, inputs, grad_outputs, dtype='d',
                **self.check_backward_options)

    def test_backward(self, backend_config):
        self.check_backward(self.inputs, self.grad_outputs, backend_config)

    def test_backward_cpu_more_than_once(self):
        func = functions.MaxPooling2D(
            3, stride=2, pad=1, cover_all=self.cover_all)
        func.apply(self.inputs)
        func.backward((0,), self.grad_outputs)
        func.backward((0,), self.grad_outputs)

    def check_double_backward(
            self, inputs, grad_outputs, grad_grad_inputs, backend_config):
        if backend_config.use_cuda:
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)
            grad_grad_inputs = cuda.to_gpu(grad_grad_inputs)
        if not self.c_contiguous:
            inputs = _to_fcontiguous(inputs)
            grad_outputs = _to_fcontiguous(grad_outputs)
            grad_grad_inputs = _to_fcontiguous(grad_grad_inputs)

        def f(x):
            y = functions.max_pooling_2d(
                x, 3, stride=2, pad=1, cover_all=self.cover_all)
            return y * y

        with backend_config:
            gradient_check.check_double_backward(
                f, inputs, grad_outputs, grad_grad_inputs,
                dtype='d',
                **self.check_double_backward_options)

    def test_double_backward(self, backend_config):
        self.check_double_backward(
            self.inputs, self.grad_outputs, self.grad_grad_inputs,
            backend_config)


@testing.parameterize(*testing.product({
    'use_cudnn': ['always', 'auto', 'never'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestMaxPooling2DCudnnCall(unittest.TestCase):

    def setUp(self):
        self.x = cuda.cupy.arange(
            2 * 3 * 4 * 3, dtype=self.dtype).reshape(2, 3, 4, 3)
        self.gy = cuda.cupy.random.uniform(-1, 1,
                                           (2, 3, 2, 2)).astype(self.dtype)

    def forward(self):
        x = chainer.Variable(self.x)
        return functions.max_pooling_2d(
            x, 3, stride=2, pad=1, cover_all=False)

    def test_call_cudnn_forward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with mock.patch('cupy.cuda.cudnn.poolingForward') as func:
                self.forward()
                self.assertEqual(func.called,
                                 chainer.should_use_cudnn('>=auto'))

    def test_call_cudnn_backward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            expect = chainer.should_use_cudnn('>=auto')
            y = self.forward()
        # should be consistent to forward regardless of use_cudnn config
        y.grad = self.gy
        with mock.patch('cupy.cuda.cudnn.poolingBackward') as func:
            y.backward()
            self.assertEqual(func.called, expect)


testing.run_module(__name__, __file__)
