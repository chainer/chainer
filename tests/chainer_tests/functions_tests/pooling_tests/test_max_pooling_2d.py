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


def _to_fcontiguous(arrays):
    xp = chainer.backend.get_array_module(*arrays)
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
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0'],
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

    def check_forward(self, inputs, backend_config):
        y_expect, = self.forward_cpu(inputs)

        # TODO(sonots): Cleanup to use testing.backend.get_array after
        # chainerx.asfortranarray is implemented.
        if (backend_config.use_cuda
            or (backend_config.use_chainerx
                and backend_config.chainerx_device.startswith('cuda:'))):
            inputs = cuda.to_gpu(inputs)
        if not self.c_contiguous:
            inputs = _to_fcontiguous(inputs)
        if backend_config.use_chainerx:
            inputs = chainer.backend.to_chx(inputs)

        with backend_config:
            x, = inputs
            y = functions.max_pooling_2d(x, 3, stride=2, pad=1,
                                         cover_all=self.cover_all)
        assert self.dtype == y.data.dtype
        assert self.output_shape == y.data.shape
        testing.assert_allclose(y_expect, y.data)

    def test_forward(self, backend_config):
        self.check_forward(self.inputs, backend_config)

    def test_forward_cpu_wide(self):  # see #120
        x_data = numpy.random.rand(2, 3, 15, 15).astype(self.dtype)
        x = chainer.Variable(x_data)
        functions.max_pooling_2d(x, 6, stride=6, pad=0)

    def test_forward_output_size_zero(self, backend_config):
        with self.assertRaises(Exception):
            x = numpy.random.rand(4, 4, 1, 4).astype(self.dtype)
            # TODO(sonots): Cleanup to use testing.backend.get_array after
            # chainerx.asfortranarray is implemented.
            if (backend_config.use_cuda
                or (backend_config.use_chainerx
                    and backend_config.chainerx_device.startswith('cuda:'))):
                x = cuda.to_gpu(x)
            if backend_config.use_chainerx:
                x = chainer.backend.to_chx(x)
            x = chainer.Variable(x)
            with backend_config:
                functions.max_pooling_2d(x, 3, stride=2)

        with self.assertRaises(Exception):
            x = numpy.random.rand(4, 4, 4, 1).astype(self.dtype)
            # TODO(sonots): Cleanup to use testing.backend.get_array after
            # chainerx.asfortranarray is implemented.
            if (backend_config.use_cuda
                or (backend_config.use_chainerx
                    and backend_config.chainerx_device.startswith('cuda:'))):
                x = cuda.to_gpu(x)
            if backend_config.use_chainerx:
                x = chainer.backend.to_chx(x)
            x = chainer.Variable(x)
            with backend_config:
                functions.max_pooling_2d(x, 3, stride=2)

    def check_backward(self, inputs, grad_outputs, backend_config):
        # TODO(sonots): Cleanup to use testing.backend.get_array after
        # chainerx.asfortranarray is implemented.
        if (backend_config.use_cuda
            or (backend_config.use_chainerx
                and backend_config.chainerx_device.startswith('cuda:'))):
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)
        if not self.c_contiguous:
            inputs = _to_fcontiguous(inputs)
            grad_outputs = _to_fcontiguous(grad_outputs)
        if backend_config.use_chainerx:
            inputs = chainer.backend.to_chx(inputs)
            grad_outputs = chainer.backend.to_chx(grad_outputs)

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
        func = functions.pooling.max_pooling_2d.MaxPooling2D(
            3, stride=2, pad=1, cover_all=self.cover_all)
        func.apply(self.inputs)
        func.backward((0,), self.grad_outputs)
        func.backward((0,), self.grad_outputs)

    def check_double_backward(
            self, inputs, grad_outputs, grad_grad_inputs, backend_config):
        # TODO(sonots): Cleanup to use testing.backend.get_array after
        # chainerx.asfortranarray is implemented.
        if (backend_config.use_cuda
            or (backend_config.use_chainerx
                and backend_config.chainerx_device.startswith('cuda:'))):
            inputs = cuda.to_gpu(inputs)
            grad_outputs = cuda.to_gpu(grad_outputs)
            grad_grad_inputs = cuda.to_gpu(grad_grad_inputs)
        if not self.c_contiguous:
            inputs = _to_fcontiguous(inputs)
            grad_outputs = _to_fcontiguous(grad_outputs)
            grad_grad_inputs = _to_fcontiguous(grad_grad_inputs)
        if backend_config.use_chainerx:
            inputs = chainer.backend.to_chx(inputs)
            grad_outputs = chainer.backend.to_chx(grad_outputs)
            grad_grad_inputs = chainer.backend.to_chx(grad_grad_inputs)

        def f(x):
            return functions.max_pooling_2d(
                x, 3, stride=2, pad=1, cover_all=self.cover_all)

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
        self.x = numpy.arange(
            2 * 3 * 4 * 4, dtype=numpy.float32).reshape(2, 3, 4, 4)
        numpy.random.shuffle(self.x)

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
