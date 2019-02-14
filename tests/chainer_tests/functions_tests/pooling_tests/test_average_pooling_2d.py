import unittest

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import backend
from chainer.testing import condition
import chainerx


def _to_fcontiguous(arrays):
    xp = chainer.backend.get_array_module(*arrays)
    # TODO(niboshi): Fix it. Non-contiguous tests are skipped for ChainerX.
    if xp is chainerx:
        raise unittest.SkipTest('ChainerX does not support asfortranarray')
    return [xp.asfortranarray(a) for a in arrays]


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'c_contiguous': [True, False],
}))
@backend.inject_backend_tests(
    ['test_forward', 'test_backward', 'test_double_backward'],
    # CPU tests
    testing.product({
        'use_cuda': [False],
        'use_ideep': ['never', 'always'],
    })
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
    })
    # ChainerX tests
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
    ])
class TestAveragePooling2D(unittest.TestCase):

    def setUp(self):
        dtype = self.dtype
        x = numpy.random.uniform(-1, 1, (2, 3, 4, 3)).astype(dtype)
        gy = numpy.random.uniform(-1, 1, (2, 3, 2, 2)).astype(dtype)
        ggx = numpy.random.uniform(-1, 1, (2, 3, 4, 3)).astype(dtype)

        self.output_shape = gy.shape

        self.inputs = [x]
        self.grad_outputs = [gy]
        self.grad_grad_inputs = [ggx]

        self.check_forward_options = {}
        self.check_backward_options = {'dtype': numpy.float64}
        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 5e-4, 'rtol': 5e-3}

    def forward_cpu(self, inputs):
        x, = inputs
        expect = numpy.empty(self.output_shape, dtype=self.dtype)
        for k in six.moves.range(2):
            for c in six.moves.range(3):
                xx = x[k, c]
                expect[k, c] = numpy.array([
                    [xx[0:2, 0:2].sum(), xx[0:2, 1:3].sum()],
                    [xx[1:4, 0:2].sum(), xx[1:4, 1:3].sum()]]) / 9
        return expect,

    def check_forward(self, inputs, backend_config):
        y_expect, = self.forward_cpu(inputs)

        inputs = backend_config.get_array(inputs)
        if not self.c_contiguous:
            with backend_config:
                inputs = _to_fcontiguous(inputs)

        with backend_config:
            x, = inputs
            y = functions.average_pooling_2d(x, 3, stride=2, pad=1)
        assert y.data.dtype == self.dtype
        y_data = cuda.to_cpu(y.data)

        assert self.output_shape == y_data.shape
        testing.assert_allclose(y_expect, y_data, **self.check_forward_options)

    @condition.retry(3)
    def test_forward(self, backend_config):
        self.check_forward(self.inputs, backend_config)

    def check_backward(self, inputs, grad_outputs, backend_config):
        inputs = backend_config.get_array(inputs)
        grad_outputs = backend_config.get_array(grad_outputs)
        if not self.c_contiguous:
            with backend_config:
                inputs = _to_fcontiguous(inputs)
                grad_outputs = _to_fcontiguous(grad_outputs)

        def f(x):
            return functions.average_pooling_2d(x, 3, 2, 1)

        with backend_config:
            gradient_check.check_backward(
                f, inputs, grad_outputs, **self.check_backward_options)

    @condition.retry(3)
    def test_backward(self, backend_config):
        self.check_backward(self.inputs, self.grad_outputs, backend_config)

    def check_double_backward(
            self, inputs, grad_outputs, grad_grad_inputs, backend_config):
        inputs = backend_config.get_array(inputs)
        grad_outputs = backend_config.get_array(grad_outputs)
        grad_grad_inputs = backend_config.get_array(grad_grad_inputs)
        if not self.c_contiguous:
            with backend_config:
                inputs = _to_fcontiguous(inputs)
                grad_outputs = _to_fcontiguous(grad_outputs)
                grad_grad_inputs = _to_fcontiguous(grad_grad_inputs)

        def f(x):
            return functions.average_pooling_2d(x, 3, 2, 1)

        with backend_config:
            gradient_check.check_double_backward(
                f, inputs, grad_outputs, grad_grad_inputs,
                **self.check_backward_options)

    @condition.retry(3)
    def test_double_backward(self, backend_config):
        self.check_double_backward(
            self.inputs, self.grad_outputs, self.grad_grad_inputs,
            backend_config)


@testing.parameterize(*testing.product({
    'use_cudnn': ['always', 'auto', 'never'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestAveragePooling2DCudnnCall(unittest.TestCase):

    def setUp(self):
        self.x = cuda.cupy.arange(
            2 * 3 * 4 * 3, dtype=self.dtype).reshape(2, 3, 4, 3)
        self.gy = cuda.cupy.random.uniform(-1, 1,
                                           (2, 3, 2, 2)).astype(self.dtype)

    def forward(self):
        x = chainer.Variable(self.x)
        return functions.average_pooling_2d(x, 3, stride=2, pad=1)

    def test_call_cudnn_forward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with testing.patch('cupy.cudnn.pooling_forward') as func:
                self.forward()
                self.assertEqual(func.called,
                                 chainer.should_use_cudnn('>=auto'))

    def test_call_cudnn_backward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            expect = chainer.should_use_cudnn('>=auto')
            y = self.forward()
        # should be consistent to forward regardless of use_cudnn config
        y.grad = self.gy
        with testing.patch('cupy.cudnn.pooling_backward') as func:
            y.backward()
            self.assertEqual(func.called, expect)


testing.run_module(__name__, __file__)
