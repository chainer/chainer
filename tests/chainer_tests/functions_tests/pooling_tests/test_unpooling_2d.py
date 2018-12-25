import unittest

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [
        # we assume insize as (2, 1)
        # standard output size which is estimated with get_deconv_outsize
        # function
        {'cover_all': False, 'outsize': (4, 2)},
        {'cover_all': True, 'outsize': (3, 1)},
        {'cover_all': False, 'outsize': None, 'expected_outsize': (4, 2)},
        {'cover_all': True, 'outsize': None, 'expected_outsize': (3, 1)},
        # another sizes which can be outsize of insize (2, 1)
        {'cover_all': False, 'outsize': (5, 2)},
        {'cover_all': True, 'outsize': (4, 2)},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
class TestUnpooling2D(unittest.TestCase):

    def setUp(self):
        self.N = 2
        self.n_channels = 3
        inh, inw = 2, 1
        self.x = numpy.arange(
            self.N * self.n_channels * inh * inw,
            dtype=self.dtype).reshape(self.N, self.n_channels, inh, inw)
        numpy.random.shuffle(self.x)
        self.x = 2 * self.x / self.x.size - 1

        self.ksize = 2
        outh, outw = self.outsize or self.expected_outsize
        self.gy = numpy.random.uniform(
            -1, 1, (self.N, self.n_channels, outh, outw)).astype(self.dtype)
        self.check_backward_options = {'atol': 1e-4, 'rtol': 1e-3}
        self.check_double_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_backward_options = {'atol': 2e-3, 'rtol': 2e-2}
            self.check_double_backward_options = {'atol': 3e-3, 'rtol': 3e-2}
        self.ggx = numpy.random.uniform(
            -1, 1, self.x.shape).astype(self.dtype)

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.unpooling_2d(x, self.ksize, outsize=self.outsize,
                                   cover_all=self.cover_all)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)

        self.assertEqual(self.gy.shape, y_data.shape)
        for i in six.moves.range(self.N):
            for c in six.moves.range(self.n_channels):
                outsize = self.outsize or self.expected_outsize
                assert y_data.shape[2:] == outsize
                if outsize == (5, 2):
                    expect = numpy.zeros(outsize, dtype=self.dtype)
                    expect[:2, :] = self.x[i, c, 0, 0]
                    expect[2:4, :] = self.x[i, c, 1, 0]
                elif outsize == (4, 2):
                    expect = numpy.array([
                        [self.x[i, c, 0, 0], self.x[i, c, 0, 0]],
                        [self.x[i, c, 0, 0], self.x[i, c, 0, 0]],
                        [self.x[i, c, 1, 0], self.x[i, c, 1, 0]],
                        [self.x[i, c, 1, 0], self.x[i, c, 1, 0]],
                    ])
                elif outsize == (3, 1):
                    expect = numpy.array([
                        [self.x[i, c, 0, 0]],
                        [self.x[i, c, 0, 0]],
                        [self.x[i, c, 1, 0]],
                    ])
                else:
                    raise ValueError('Unsupported outsize: {}'.format(outsize))
                testing.assert_allclose(expect, y_data[i, c])

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        def f(x):
            return functions.unpooling_2d(x, self.ksize, outsize=self.outsize,
                                          cover_all=self.cover_all)
        gradient_check.check_backward(
            f, x_data, y_grad, dtype=numpy.float64,
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, x_grad_grad,
                              use_cudnn='always'):
        def f(x):
            return functions.unpooling_2d(x, self.ksize, outsize=self.outsize,
                                          cover_all=self.cover_all)
        with chainer.using_config('use_cudnn', use_cudnn):
            gradient_check.check_double_backward(
                f, x_data, y_grad, x_grad_grad, dtype=numpy.float64,
                **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(
            self.x, self.gy, self.ggx, 'never')

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))

    @attr.gpu
    def test_double_backward_gpu_non_contiguous(self):
        self.check_double_backward(
            cuda.cupy.asfortranarray(cuda.to_gpu(self.x)),
            cuda.cupy.asfortranarray(cuda.to_gpu(self.gy)),
            cuda.cupy.asfortranarray(cuda.to_gpu(self.ggx)))

    @attr.gpu
    def test_double_backward_gpu_no_cudnn(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx),
            'never')


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'h': [5],
    'k': [3],
    's': [3],
    'p': [0],
    'cover_all': [True, False],
}))
class TestMaxPoolingUnpooling(unittest.TestCase):

    def check_left_inverse(self, xp, use_cudnn='never'):
        x = xp.arange(self.h * self.h).reshape(
            (1, 1, self.h, self.h)).astype(self.dtype)
        with chainer.using_config('use_cudnn', use_cudnn):
            y = chainer.functions.unpooling_2d(
                x, self.k, self.s, self.p, None, self.cover_all)
            x_ = chainer.functions.max_pooling_2d(
                y, self.k, self.s, self.p, self.cover_all).data

        self.assertEqual(x.shape, x_.shape)
        self.assertEqual(x.dtype, x_.dtype)
        chainer.testing.assert_allclose(x, x_)

    def test_left_inverse_cpu(self):
        self.check_left_inverse(numpy)

    @attr.gpu
    def test_left_inverse_cupy(self):
        self.check_left_inverse(cuda.cupy)

    @attr.gpu
    def test_left_inverse_cudnn(self):
        self.check_left_inverse(cuda.cupy, 'always')


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'h': [5],
    'k': [3],
    's': [3],
    'p': [0],
}))
class TestAveragePoolingUnpooling(unittest.TestCase):

    def check_left_inverse(self, xp, use_cudnn='never'):
        x = xp.arange(self.h * self.h).reshape(
            (1, 1, self.h, self.h)).astype(self.dtype)
        with chainer.using_config('use_cudnn', use_cudnn):
            # average_pooling_2d does not have cover_all option
            # as max_pooling_2d has.
            y = chainer.functions.unpooling_2d(
                x, self.k, self.s, self.p, None, False)
            x_ = chainer.functions.average_pooling_2d(
                y, self.k, self.s, self.p).data

        self.assertEqual(x.shape, x_.shape)
        self.assertEqual(x.dtype, x_.dtype)
        chainer.testing.assert_allclose(x, x_)

    def test_left_inverse_cpu(self):
        self.check_left_inverse(numpy)

    @attr.gpu
    def test_left_inverse_cupy(self):
        self.check_left_inverse(cuda.cupy)

    @attr.gpu
    def test_left_inverse_cudnn(self):
        self.check_left_inverse(cuda.cupy, 'always')


testing.run_module(__name__, __file__)
