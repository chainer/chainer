import unittest

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.testing import backend
from chainer_tests.functions_tests.pooling_tests import pooling_nd_helper


_inject_backend_tests = backend.inject_backend_tests(
    None,
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
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0'],
    })
)


@_inject_backend_tests
@testing.parameterize(*testing.product({
    'cover_all': [True, False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'contiguous': [None, 'C'],
}))
class TestMaxPooling2D(testing.FunctionTestCase):

    def setUp(self):
        if self.cover_all:
            self.output_shape = (2, 3, 3, 2)
        else:
            self.output_shape = (2, 3, 2, 2)

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

    def generate_inputs(self):
        return pooling_nd_helper.shuffled_linspace((2, 3, 4, 3), self.dtype),

    def forward_expected(self, inputs):
        x, = inputs
        expect = numpy.empty(self.output_shape, dtype=self.dtype)
        for i in six.moves.range(2):
            for c in six.moves.range(3):
                xx = x[i, c]
                if self.cover_all:
                    expect[i, c] = numpy.array([
                        [xx[0:2, 0:2].max(), xx[0:2, 1:3].max()],
                        [xx[1:4, 0:2].max(), xx[1:4, 1:3].max()],
                        [xx[3:4, 0:2].max(), xx[3:4, 1:3].max()]])
                else:
                    expect[i, c] = numpy.array([
                        [xx[0:2, 0:2].max(), xx[0:2, 1:3].max()],
                        [xx[1:4, 0:2].max(), xx[1:4, 1:3].max()]])
        return expect,

    def forward(self, inputs, device):
        x, = inputs
        y = functions.max_pooling_2d(x, 3, stride=2, pad=1,
                                     cover_all=self.cover_all)
        return y,


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestMaxPooling2DForwardCpuWide(unittest.TestCase):
    # see #120

    def test_forward_cpu_wide(self):
        x_data = numpy.random.rand(2, 3, 15, 15).astype(self.dtype)
        x = chainer.Variable(x_data)
        functions.max_pooling_2d(x, 6, stride=6, pad=0)


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


class TestMaxPooling2DIndices(unittest.TestCase):
    def setUp(self):
        self.x = pooling_nd_helper.shuffled_linspace(
            (2, 3, 4, 4), numpy.float32)

    def _check(self, x):
        out, indices = functions.max_pooling_2d(
            x, 2, cover_all=False, return_indices=True)
        assert isinstance(out, chainer.Variable)
        assert isinstance(out.array, type(x))
        assert isinstance(indices, type(x))
        assert indices.shape == out.array.shape

        # Calculate expected indices.
        expect = numpy.zeros(indices.shape, dtype=indices.dtype)
        for i in six.moves.range(2):
            for c in six.moves.range(3):
                xx = x[i, c]
                expect[i, c] = numpy.array([
                    [xx[0:2, 0:2].ravel().argmax(),
                     xx[0:2, 2:4].ravel().argmax()],
                    [xx[2:4, 0:2].ravel().argmax(),
                     xx[2:4, 2:4].ravel().argmax()],
                ])
        if out.xp is cuda.cupy:
            expect = cuda.to_gpu(expect)
        assert (expect == indices).all()

    def test_cpu(self):
        self._check(self.x)

    @attr.gpu
    @attr.cudnn
    def test_gpu(self):
        x = cuda.to_gpu(self.x)
        with chainer.using_config('use_cudnn', 'never'):
            self._check(x)
        with chainer.using_config('use_cudnn', 'always'):
            self._check(x)


testing.run_module(__name__, __file__)
