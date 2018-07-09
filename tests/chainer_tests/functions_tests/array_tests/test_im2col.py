import unittest

import numpy
from six import moves

from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr

from chainer.utils.conv import get_conv_outsize


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


@testing.parameterize(*testing.product_dict(
    [
        {'params': (1, 1, 1, 1, 1, 1, 1, 1)},
        {'params': (2, 2, 2, 2, 2, 2, 2, 2)},
        {'params': (1, 2, 2, 1, 1, 2, 1, 1)},
        {'params': (1, 2, 3, 4, 1, 2, 1, 1)},
        {'params': (1, 2, 3, 4, 4, 5, 2, 3)},
        {'params': (3, 3, 2, 2, 1, 1, 1, 1)}
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64}
    ]
))
class TestIm2ColForward(unittest.TestCase):

    in_shape = (2, 3, 8, 6)

    def setUp(self):
        self.x = numpy.random.uniform(
            size=self.in_shape).astype(self.dtype)

    def check_forward(self, x, kh, kw, sy, sx, ph, pw, dy, dx, gpu):
        x = x.copy()
        n, c, h, w = x.shape
        col = functions.im2col(
            x, (kh, kw), (sy, sx), (ph, pw), dilate=(dy, dx)).data
        col_h = get_conv_outsize(h, kh, sy, ph, d=dy)
        col_w = get_conv_outsize(w, kw, sx, pw, d=dx)

        self.assertEqual(col.shape, (n, c * kh * kw, col_h, col_w))
        col = col.reshape(n, c, kh, kw, col_h, col_w)
        col = cuda.to_cpu(col)

        for y in moves.range(col_h):
            for x in moves.range(col_w):
                for ky in moves.range(kh):
                    for kx in moves.range(kw):
                        oy = y * sy - ph + ky * dy
                        ox = x * sx - pw + kx * dx
                        if 0 <= oy < h and 0 <= ox < w:
                            testing.assert_allclose(
                                col[:, :, ky, kx, y, x],
                                self.x[:, :, oy, ox])
                        else:
                            testing.assert_allclose(
                                col[:, :, ky, kx, y, x],
                                numpy.zeros((2, 3), self.dtype))

    def test_forward_cpu(self):
        self.check_forward(self.x, *self.params, gpu=False)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), *self.params, gpu=True)


@testing.parameterize(*testing.product_dict(
    [
        {'ksize': 1, 'stride': 1, 'pad': 0, 'dilate': 1},
        {'ksize': (1, 1), 'stride': (1, 1), 'pad': (1, 0), 'dilate': (1, 1)},
        {'ksize': 2, 'stride': 2, 'pad': 2, 'dilate': 2},
        {'ksize': (2, 3), 'stride': (1, 2), 'pad': 0, 'dilate': (2, 1)},
    ],
    [
        {'cover_all': False},
        {'cover_all': True},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ]
))
class TestIm2Col(unittest.TestCase):

    in_shape = (2, 3, 8, 6)

    def setUp(self):
        self.x = numpy.random.uniform(
            size=self.in_shape).astype(self.dtype)

        kh, kw = _pair(self.ksize)
        sy, sx = _pair(self.stride)
        ph, pw = _pair(self.pad)
        dy, dx = _pair(self.dilate)

        N, C, H, W = self.in_shape

        o_H = get_conv_outsize(H, kh, sy, ph, cover_all=self.cover_all, d=dy)
        o_W = get_conv_outsize(W, kw, sx, pw, cover_all=self.cover_all, d=dx)

        self.gy = numpy.random.uniform(
            size=(N, C * kh * kw, o_H, o_W)).astype(self.dtype)
        self.ggx = numpy.random.uniform(
            size=self.in_shape).astype(self.dtype)

        self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}
        if self.dtype is numpy.float16:
            self.check_backward_options.update({'atol': 1e-3, 'rtol': 1e-2})

        self.check_double_backward_options = {'atol': 5e-4, 'rtol': 5e-3}
        if self.dtype is numpy.float16:
            self.check_double_backward_options.update(
                {'atol': 1e-3, 'rtol': 1e-2})

    def check_backward(self, x, ksize, stride, pad, cover_all, dilate, gy):
        def f(x):
            return functions.im2col(
                x, ksize, stride=stride, pad=pad, cover_all=cover_all,
                dilate=dilate)

        gradient_check.check_backward(
            f, x, gy, dtype=numpy.float64, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(
            self.x, self.ksize, self.stride, self.pad, self.cover_all,
            self.dilate, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), self.ksize, self.stride, self.pad,
            self.cover_all, self.dilate, cuda.to_gpu(self.gy))

    def check_double_backward(self, x, ksize, stride, pad, cover_all, dilate,
                              gy, ggx):
        def f(x):
            return functions.im2col(
                x, ksize, stride=stride, pad=pad, cover_all=cover_all,
                dilate=dilate)

        gradient_check.check_double_backward(
            f, x, gy, ggx, dtype=numpy.float64,
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(
            self.x, self.ksize, self.stride, self.pad, self.cover_all,
            self.dilate, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x),
            self.ksize, self.stride, self.pad, self.cover_all, self.dilate,
            cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))


testing.run_module(__name__, __file__)
