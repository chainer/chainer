import operator
import sys
import unittest

import numpy
import pytest

import chainer
from chainer.backends import cuda
from chainer import basic_math
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import backend
from chainer.utils import type_check
import chainerx


def arrays_to_chainerx(orig_xp, np_arrays):
    assert all(isinstance(a, numpy.ndarray) for a in np_arrays)
    if orig_xp is numpy:
        orig_arrays = np_arrays
    elif orig_xp is cuda.cupy:
        orig_arrays = [cuda.to_gpu(a) for a in np_arrays]
    return [chainer.backend.to_chainerx(a) for a in orig_arrays]


@testing.parameterize(*testing.product({
    'shape': [
        # x1, x2, y
        ((3, 2), (3, 2), (3, 2)),
        ((), (), ()),
        ((3, 2), (3, 1), (3, 2)),
        ((2,), (3, 2), (3, 2)),
    ],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestBinaryOp(unittest.TestCase):

    def setUp(self):
        self.x1 = numpy.random.uniform(.5, 1, self.shape[0]).astype(self.dtype)
        self.x2 = numpy.random.uniform(.5, 1, self.shape[1]).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, self.shape[2]).astype(self.dtype)
        self.ggx1 = numpy.random.uniform(-1, 1, self.shape[0]).astype(
            self.dtype)
        self.ggx2 = numpy.random.uniform(-1, 1, self.shape[1]).astype(
            self.dtype)

    def check_forward(self, op, x1_data, x2_data):
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        y = op(x1, x2)
        options = {}
        if self.dtype == numpy.float16:
            options = {'atol': 1e-4, 'rtol': 1e-3}
        testing.assert_allclose(op(self.x1, self.x2), y.data, **options)

    def forward_cpu(self, op):
        self.check_forward(op, self.x1, self.x2)

    def test_add_forward_cpu(self):
        self.forward_cpu(lambda x, y: x + y)

    def test_sub_forward_cpu(self):
        self.forward_cpu(lambda x, y: x - y)

    def test_mul_forward_cpu(self):
        self.forward_cpu(lambda x, y: x * y)

    def test_div_forward_cpu(self):
        self.forward_cpu(lambda x, y: x / y)

    def test_floordiv_forward_cpu(self):
        self.forward_cpu(lambda x, y: x // y)

    def test_pow_forward_cpu(self):
        self.forward_cpu(lambda x, y: x ** y)

    def test_radd_forward_cpu(self):
        self.forward_cpu(lambda x, y: y.__radd__(x))

    def test_rsub_forward_cpu(self):
        self.forward_cpu(lambda x, y: y.__rsub__(x))

    def test_rmul_forward_cpu(self):
        self.forward_cpu(lambda x, y: y.__rmul__(x))

    def test_rdiv_forward_cpu(self):
        self.forward_cpu(lambda x, y: y.__rtruediv__(x))

    def test_rfloordiv_forward_cpu(self):
        self.forward_cpu(lambda x, y: y.__rfloordiv__(x))

    def test_rpow_forward_cpu(self):
        self.forward_cpu(lambda x, y: y.__rpow__(x))

    def forward_gpu(self, op):
        self.check_forward(op, cuda.to_gpu(self.x1), cuda.to_gpu(self.x2))

    @attr.gpu
    def test_add_forward_gpu(self):
        self.forward_gpu(lambda x, y: x + y)

    @attr.gpu
    def test_sub_forward_gpu(self):
        self.forward_gpu(lambda x, y: x - y)

    @attr.gpu
    def test_mul_forward_gpu(self):
        self.forward_gpu(lambda x, y: x * y)

    @attr.gpu
    def test_div_forward_gpu(self):
        self.forward_gpu(lambda x, y: x / y)

    @attr.gpu
    def test_floordiv_forward_gpu(self):
        self.forward_gpu(lambda x, y: x // y)

    @attr.gpu
    def test_pow_forward_gpu(self):
        self.forward_gpu(lambda x, y: x ** y)

    @attr.gpu
    def test_radd_forward_gpu(self):
        self.forward_gpu(lambda x, y: y.__radd__(x))

    @attr.gpu
    def test_rsub_forward_gpu(self):
        self.forward_gpu(lambda x, y: y.__rsub__(x))

    @attr.gpu
    def test_rmul_forward_gpu(self):
        self.forward_gpu(lambda x, y: y.__rmul__(x))

    @attr.gpu
    def test_rdiv_forward_gpu(self):
        self.forward_gpu(lambda x, y: y.__rtruediv__(x))

    @attr.gpu
    def test_rfloordiv_forward_gpu(self):
        self.forward_gpu(lambda x, y: y.__rfloordiv__(x))

    @attr.gpu
    def test_rpow_forward_gpu(self):
        self.forward_gpu(lambda x, y: y.__rpow__(x))

    @attr.gpu
    def test_add_constant_allocation(self):
        x = 0
        y = chainer.Variable(cuda.cupy.ones((1,)))
        z = y + x
        self.assertEqual(1, z.data.get()[0])

    def forward_chainerx(self, op, orig_xp):
        xs_chx = arrays_to_chainerx(orig_xp, (self.x1, self.x2))
        self.check_forward(op, *xs_chx)

    @attr.chainerx
    def test_add_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: x + y, numpy)

    @attr.chainerx
    def test_sub_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: x - y, numpy)

    @attr.chainerx
    def test_mul_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: x * y, numpy)

    @attr.chainerx
    def test_div_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: x / y, numpy)

    @attr.chainerx
    @attr.gpu
    def test_add_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: x + y, cuda.cupy)

    @attr.chainerx
    @attr.gpu
    def test_sub_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: x - y, cuda.cupy)

    @attr.chainerx
    @attr.gpu
    def test_mul_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: x * y, cuda.cupy)

    @attr.chainerx
    @attr.gpu
    def test_div_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: x / y, cuda.cupy)

    # TODO(hvy): Implement floor.
    @pytest.mark.skip
    @attr.chainerx
    def test_floordiv_forward_chainerx_cpu(self):
        pass

    @attr.chainerx
    def test_pow_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: x.__pow__(y), numpy)

    @attr.chainerx
    def test_radd_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: y.__radd__(x), numpy)

    @attr.chainerx
    def test_rsub_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: y.__rsub__(x), numpy)

    @attr.chainerx
    def test_rmul_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: y.__rmul__(x), numpy)

    @attr.chainerx
    def test_rdiv_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: y.__rtruediv__(x), numpy)

    # TODO(hvy): Implement floor.
    @pytest.mark.skip
    @attr.chainerx
    def test_rfloordiv_forward_chainerx_cpu(self):
        pass

    @attr.chainerx
    def test_rpow_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: y.__rpow__(x), numpy)

    # TODO(hvy): Implement floor.
    @pytest.mark.skip
    @attr.chainerx
    @attr.gpu
    def test_floordiv_forward_chainerx_gpu(self):
        pass

    @attr.chainerx
    @attr.gpu
    def test_pow_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: x.__pow__(y), cuda.cupy)

    @attr.chainerx
    @attr.gpu
    def test_radd_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: y.__radd__(x), cuda.cupy)

    @attr.chainerx
    @attr.gpu
    def test_rsub_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: y.__rsub__(x), cuda.cupy)

    @attr.chainerx
    @attr.gpu
    def test_rmul_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: y.__rmul__(x), cuda.cupy)

    @attr.chainerx
    @attr.gpu
    def test_rdiv_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: y.__rtruediv__(x), cuda.cupy)

    # TODO(hvy): Implement floor.
    @pytest.mark.skip
    @attr.chainerx
    @attr.gpu
    def test_rfloordiv_forward_chainerx_gpu(self):
        pass

    @attr.chainerx
    @attr.gpu
    def test_rpow_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: y.__rpow__(x), cuda.cupy)

    def check_backward(self, op, x1_data, x2_data, y_grad):
        options = {}
        if self.dtype == numpy.float16:
            options = {'atol': 5e-3, 'rtol': 5e-2}
        gradient_check.check_backward(op, (x1_data, x2_data), y_grad,
                                      dtype=numpy.float64, **options)

    def backward_cpu(self, op):
        self.check_backward(op, self.x1, self.x2, self.gy)

    def test_add_backward_cpu(self):
        self.backward_cpu(lambda x, y: x + y)

    def test_sub_backward_cpu(self):
        self.backward_cpu(lambda x, y: x - y)

    def test_mul_backward_cpu(self):
        self.backward_cpu(lambda x, y: x * y)

    def test_div_backward_cpu(self):
        self.backward_cpu(lambda x, y: x / y)

    def test_pow_backward_cpu(self):
        self.backward_cpu(lambda x, y: x ** y)

    def backward_gpu(self, op):
        self.check_backward(
            op, cuda.to_gpu(self.x1), cuda.to_gpu(self.x2),
            cuda.to_gpu(self.gy))

    @attr.gpu
    def test_add_backward_gpu(self):
        self.backward_gpu(lambda x, y: x + y)

    @attr.gpu
    def test_sub_backward_gpu(self):
        self.backward_gpu(lambda x, y: x - y)

    @attr.gpu
    def test_mul_backward_gpu(self):
        self.backward_gpu(lambda x, y: x * y)

    @attr.gpu
    def test_div_backward_gpu(self):
        self.backward_gpu(lambda x, y: x / y)

    @attr.gpu
    def test_pow_backward_gpu(self):
        self.backward_gpu(lambda x, y: x ** y)

    def backward_chainerx(self, op):
        self.check_backward(
            op, chainerx.array(self.x1), chainerx.array(self.x2),
            chainerx.array(self.gy))

    @attr.chainerx
    def test_add_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: x + y)

    @attr.chainerx
    def test_sub_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: x - y)

    @attr.chainerx
    def test_mul_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: x * y)

    @attr.chainerx
    def test_div_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: x / y)

    @attr.chainerx
    def test_pow_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: x ** y)

    def check_double_backward(
            self, op, x1_data, x2_data, y_grad, ggx1_data, ggx2_data, **args):
        options = {}
        if self.dtype == numpy.float16:
            options = {'atol': 5e-3, 'rtol': 5e-2}
        options.update(args)

        gradient_check.check_double_backward(
            op, (x1_data, x2_data), y_grad, (ggx1_data, ggx2_data),
            dtype=numpy.float64, **options)

    def double_backward_cpu(self, op, **options):
        self.check_double_backward(
            op, self.x1, self.x2, self.gy, self.ggx1, self.ggx2,
            **options)

    def test_div_double_backward_cpu(self):
        self.double_backward_cpu(lambda x, y: x / y, atol=5e-2, rtol=5e-2)

    def test_pow_double_backward_cpu(self):
        self.double_backward_cpu(lambda x, y: x ** y)

    def test_rpow_double_backward_cpu(self):
        self.double_backward_cpu(lambda x, y: y.__rpow__(x))

    def double_backward_gpu(self, op, **options):
        self.check_double_backward(
            op, cuda.to_gpu(self.x1), cuda.to_gpu(self.x2),
            cuda.to_gpu(self.gy),
            cuda.to_gpu(self.ggx1), cuda.to_gpu(self.ggx2), **options)

    @attr.gpu
    def test_div_double_backward_gpu(self):
        self.double_backward_gpu(lambda x, y: x / y, atol=5e-2, rtol=5e-2)

    @attr.gpu
    def test_pow_double_backward_gpu(self):
        self.double_backward_gpu(lambda x, y: x ** y)

    @attr.gpu
    def test_rpow_double_backward_gpu(self):
        self.double_backward_gpu(lambda x, y: y.__rpow__(x))

    def double_backward_chainerx(self, op, **options):
        self.check_double_backward(
            op, chainerx.array(self.x1), chainerx.array(self.x2),
            chainerx.array(self.gy),
            chainerx.array(self.ggx1), chainerx.array(self.ggx2), **options)

    @attr.chainerx
    def test_div_double_backward_chainerx(self):
        self.double_backward_chainerx(lambda x, y: x / y, atol=5e-2, rtol=5e-2)

    @attr.chainerx
    def test_pow_double_backward_chainerx(self):
        self.double_backward_chainerx(lambda x, y: x ** y)

    @attr.chainerx
    def test_rpow_double_backward_chainerx(self):
        self.double_backward_chainerx(lambda x, y: y.__rpow__(x))


@testing.parameterize(*testing.product({
    'in_shapes': [
        ((3, 2),) * 3,
        ((),) * 3,
        ((1, 3), (), (2, 1, 2, 1)),
        ((), (2, 1, 2), (3, 1)),
        ((3, 1), (1, 1), (2,)),
    ],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
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
    }))
class TestMultipleAdd(unittest.TestCase):

    def setUp(self):
        x1_shape, x2_shape, x3_shape = self.in_shapes
        self.x1 = numpy.random.uniform(.5, 1, x1_shape).astype(self.dtype)
        self.x2 = numpy.random.uniform(.5, 1, x2_shape).astype(self.dtype)
        self.x3 = numpy.random.uniform(.5, 1, x3_shape).astype(self.dtype)
        y_shape = numpy.broadcast(self.x1, self.x2, self.x3).shape
        self.gy = numpy.random.uniform(-1, 1, y_shape).astype(self.dtype)
        self.ggx1 = numpy.random.uniform(-1, 1, x1_shape).astype(self.dtype)
        self.ggx2 = numpy.random.uniform(-1, 1, x2_shape).astype(self.dtype)
        self.ggx3 = numpy.random.uniform(-1, 1, x3_shape).astype(self.dtype)

    def check_forward(self, func, x1_data, x2_data, x3_data, backend_config):
        # convert to cupy.ndarray for GPU tests
        if backend_config.use_cuda:
            x1_data, x2_data, x3_data = cuda.to_gpu(
                (x1_data, x2_data, x3_data))
        x1 = chainer.Variable(x1_data)
        x2 = chainer.Variable(x2_data)
        x3 = chainer.Variable(x3_data)
        with backend_config:
            y = func(x1, x2, x3)
        options = {}
        if self.dtype == numpy.float16:
            options = {'atol': 1e-4, 'rtol': 1e-3}
        testing.assert_allclose(
            (self.x1 + self.x2 + self.x3), y.data, **options)

    def forward_cpu(self, func, backend_config):
        self.check_forward(func, self.x1, self.x2, self.x3, backend_config)

    def test_forward(self, backend_config):
        func = chainer.functions.add
        self.forward_cpu(func, backend_config)

    def check_backward(self, func, x1_data, x2_data, x3_data, y_grad,
                       backend_config):
        # convert to cupy.ndarray for GPU tests
        if backend_config.use_cuda:
            x1_data, x2_data, x3_data, y_grad = cuda.to_gpu(
                (x1_data, x2_data, x3_data, y_grad))
        options = {}
        if self.dtype == numpy.float16:
            options = {'atol': 5e-3, 'rtol': 5e-2}
        with backend_config:
            gradient_check.check_backward(func, (x1_data, x2_data, x3_data),
                                          y_grad,
                                          dtype=numpy.float64, **options)

    def backward_cpu(self, func, backend_config):
        self.check_backward(
            func, self.x1, self.x2, self.x3, self.gy, backend_config)

    def test_backward(self, backend_config):
        func = chainer.functions.add
        self.backward_cpu(func, backend_config)

    def check_double_backward(
            self, func, backend_config, x1_data, x2_data, x3_data, y_grad,
            ggx1_data, ggx2_data, ggx3_data, **args):
        # convert to cupy.ndarray for GPU tests
        if backend_config.use_cuda:
            (x1_data, x2_data, x3_data, y_grad,
                ggx1_data, ggx2_data, ggx3_data) = cuda.to_gpu(
                (x1_data, x2_data, x3_data, y_grad,
                    ggx1_data, ggx2_data, ggx3_data))
        options = {}
        if self.dtype == numpy.float16:
            options = {'atol': 5e-3, 'rtol': 5e-2}
        options.update(args)

        with backend_config:
            gradient_check.check_double_backward(
                func, (x1_data, x2_data, x3_data), y_grad,
                (ggx1_data,
                 ggx2_data, ggx3_data),
                dtype=numpy.float64, **options)

    def double_backward_cpu(self, func, backend_config, **options):
        self.check_double_backward(
            func, backend_config, self.x1, self.x2, self.x3, self.gy,
            self.ggx1, self.ggx2, self.ggx3,
            **options)

    def test_double_backward(self, backend_config):
        func = chainer.functions.add
        self.double_backward_cpu(func, backend_config, atol=5e-2, rtol=5e-2)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestBinaryOpConstant(unittest.TestCase):

    def _test_constant_one(self, func, lhs, rhs, gpu=False):
        if gpu:
            lhs = cuda.to_gpu(lhs)
        x = chainer.Variable(lhs)
        y = func(x, rhs)
        self.assertEqual(y.data.dtype, self.dtype)
        y.backward()
        self.assertEqual(x.grad.dtype, self.dtype)

    def _test_constant(self, func):
        x_data = numpy.array(1, self.dtype)

        self._test_constant_one(func, x_data, 1)
        self._test_constant_one(func, x_data, 1.0)
        self._test_constant_one(func, x_data, numpy.int64(1))
        self._test_constant_one(func, x_data, numpy.float64(1.0))

    def _test_constant_gpu(self, func):
        x_data = numpy.array(1, self.dtype)

        self._test_constant_one(func, x_data, 1, True)
        self._test_constant_one(func, x_data, 1.0, True)
        self._test_constant_one(func, x_data, numpy.int64(1), True)
        self._test_constant_one(func, x_data, numpy.float64(1), True)

    def _test_constant_array_one(self, func, lhs, rhs):
        x = chainer.Variable(lhs)
        y = func(x, rhs)
        self.assertEqual(y.data.dtype, self.dtype)
        y.grad = numpy.ones_like(y.data, self.dtype)
        y.backward()
        self.assertEqual(x.grad.dtype, self.dtype)

    def _test_constant_array(self, func):
        x_data = numpy.array([1.0, 2.0], self.dtype)

        self._test_constant_array_one(
            func, x_data, numpy.array([3.0, 4.0], self.dtype))

        with pytest.raises(TypeError):
            self._test_constant_array_one(func, x_data, [3.0, 4.0])
        with pytest.raises(TypeError):
            self._test_constant_array_one(func, x_data, (3.0, 4.0))

        with pytest.raises(TypeError):
            self._test_constant_array_one(func, x_data, [3.0, 4.0, 5.0])
        with pytest.raises(TypeError):
            self._test_constant_array_one(func, x_data, (3.0, 4.0, 5.0))
        with pytest.raises(type_check.InvalidType):
            self._test_constant_array_one(
                func, x_data, numpy.array([3.0, 4.0, 5.0], self.dtype))

    def _test_constant_array_gpu_one(self, func, lhs, rhs):
        x = chainer.Variable(cuda.to_gpu(lhs))
        y = func(x, rhs)
        self.assertEqual(y.data.dtype, self.dtype)
        y.grad = cuda.cupy.ones_like(y.data).astype(self.dtype)
        y.backward()
        self.assertEqual(x.grad.dtype, self.dtype)

    def _test_constant_array_gpu(self, func, exception=TypeError):
        x_data = numpy.array([1.0, 2.0], self.dtype)

        self._test_constant_array_gpu_one(
            func, x_data, cuda.to_gpu(numpy.array([3.0, 4.0], self.dtype)))

        with pytest.raises(exception):
            self._test_constant_array_one(
                func, x_data, cuda.to_gpu(
                    numpy.array([3.0, 4.0, 5.0], self.dtype)))

    def test_add_constant(self):
        self._test_constant(lambda x, y: x + y)

    @attr.gpu
    def test_add_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: x + y)

    def test_add_constant_array(self):
        self._test_constant_array(lambda x, y: x + y)

    @attr.gpu
    def test_add_constant_array_gpu(self):
        self._test_constant_array_gpu(lambda x, y: x + y)

    def test_radd_constant(self):
        self._test_constant(lambda x, y: y + x)

    @attr.gpu
    def test_radd_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: y + x)

    def test_radd_constant_array(self):
        self._test_constant_array(lambda x, y: y + x)

    @attr.gpu
    def test_radd_constant_array_gpu(self):
        self._test_constant_array_gpu(lambda x, y: y + x)

    def test_sub_constant(self):
        self._test_constant(lambda x, y: x - y)

    @attr.gpu
    def test_sub_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: x - y)

    def test_sub_constant_array(self):
        self._test_constant_array(lambda x, y: x - y)

    @attr.gpu
    def test_sub_constant_array_gpu(self):
        self._test_constant_array_gpu(lambda x, y: x - y)

    def test_rsub_constant(self):
        self._test_constant(lambda x, y: y - x)

    @attr.gpu
    def test_rsub_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: y - x)

    def test_rsub_constant_array(self):
        self._test_constant_array(lambda x, y: y - x)

    @attr.gpu
    def test_rsub_constant_array_gpu(self):
        self._test_constant_array_gpu(lambda x, y: y - x)

    def test_mul_constant(self):
        self._test_constant(lambda x, y: x * y)

    @attr.gpu
    def test_mul_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: x * y)

    def test_mul_constant_array(self):
        self._test_constant_array(lambda x, y: x * y)

    @attr.gpu
    def test_mul_constant_array_gpu(self):
        self._test_constant_array(lambda x, y: x * y)

    def test_rmul_constant(self):
        self._test_constant(lambda x, y: y * x)

    @attr.gpu
    def test_rmul_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: y * x)

    def test_rmul_constant_array(self):
        self._test_constant_array(lambda x, y: y * x)

    @attr.gpu
    def test_rmul_constant_array_gpu(self):
        # _test_constant_array_one throws pycuda._pvt_struct.error
        self._test_constant_array_gpu(lambda x, y: y * x, exception=Exception)

    def test_div_constant(self):
        self._test_constant(lambda x, y: x / y)

    @attr.gpu
    def test_div_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: x / y)

    def test_div_constant_array(self):
        self._test_constant_array(lambda x, y: x / y)

    @attr.gpu
    def test_div_constant_array_gpu(self):
        # _test_constant_array_one throws pycuda._pvt_struct.error
        self._test_constant_array_gpu(lambda x, y: x / y, exception=Exception)

    def test_rdiv_constant(self):
        self._test_constant(lambda x, y: y / x)

    @attr.gpu
    def test_rdiv_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: y / x)

    def test_rdiv_constant_array(self):
        self._test_constant_array(lambda x, y: y / x)

    @attr.gpu
    def test_rdiv_constant_array_gpu(self):
        self._test_constant_array_gpu(lambda x, y: y / x)

    def test_pow_constant(self):
        self._test_constant(lambda x, y: x ** y)

    @attr.gpu
    def test_pow_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: x ** y)

    def test_pow_constant_array(self):
        self._test_constant_array(lambda x, y: x ** y)

    @attr.gpu
    def test_pow_constant_array_gpu(self):
        self._test_constant_array_gpu(lambda x, y: x ** y, exception=TypeError)

    def test_rpow_constant(self):
        self._test_constant(lambda x, y: y ** x)

    @attr.gpu
    def test_rpow_constant_gpu(self):
        self._test_constant_gpu(lambda x, y: y ** x)

    def test_rpow_constant_array(self):
        self._test_constant_array(lambda x, y: y ** x)

    @attr.gpu
    def test_rpow_constant_array_gpu(self):
        # _test_constant_array_one throws pycuda._pvt_struct.error
        self._test_constant_array_gpu(lambda x, y: y ** x, exception=Exception)


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestVariableConstantOp(unittest.TestCase):

    def make_date(self):
        raise NotImplementedError()

    def setUp(self):
        self.x = numpy.random.uniform(.5, 1, self.shape).astype(self.dtype)
        self.gy = numpy.random.uniform(.5, 1, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(.5, 1, self.shape).astype(self.dtype)
        self.value = 0.5

    def check_forward(self, op, x_data):
        x = chainer.Variable(x_data)
        y = op(x, self.value)
        if self.dtype == numpy.float16:
            atol = 5e-4
            rtol = 5e-4
        else:
            atol = 1e-7
            rtol = 1e-7
        testing.assert_allclose(
            op(self.x, self.value), y.data, atol=atol, rtol=rtol)

    def forward_cpu(self, op):
        self.check_forward(op, self.x)

    def test_add_forward_cpu(self):
        self.forward_cpu(lambda x, y: x + y)

    def test_radd_forward_cpu(self):
        self.forward_cpu(lambda x, y: y + x)

    def test_sub_forward_cpu(self):
        self.forward_cpu(lambda x, y: x - y)

    def test_rsub_forward_cpu(self):
        self.forward_cpu(lambda x, y: y - x)

    def test_mul_forward_cpu(self):
        self.forward_cpu(lambda x, y: x * y)

    def test_rmul_forward_cpu(self):
        self.forward_cpu(lambda x, y: y * x)

    def test_div_forward_cpu(self):
        self.forward_cpu(lambda x, y: x / y)

    def test_rdiv_forward_cpu(self):
        self.forward_cpu(lambda x, y: y / x)

    def test_pow_forward_cpu(self):
        self.forward_cpu(lambda x, y: x ** y)

    def test_rpow_forward_cpu(self):
        self.forward_cpu(lambda x, y: y ** x)

    def forward_gpu(self, op):
        self.check_forward(op, cuda.to_gpu(self.x))

    @attr.gpu
    def test_add_forward_gpu(self):
        self.forward_gpu(lambda x, y: x + y)

    @attr.gpu
    def test_radd_forward_gpu(self):
        self.forward_gpu(lambda x, y: y + x)

    @attr.gpu
    def test_sub_forward_gpu(self):
        self.forward_gpu(lambda x, y: x - y)

    @attr.gpu
    def test_rsub_forward_gpu(self):
        self.forward_gpu(lambda x, y: y - x)

    @attr.gpu
    def test_mul_forward_gpu(self):
        self.forward_gpu(lambda x, y: x * y)

    @attr.gpu
    def test_rmul_forward_gpu(self):
        self.forward_gpu(lambda x, y: y * x)

    @attr.gpu
    def test_div_forward_gpu(self):
        self.forward_gpu(lambda x, y: x / y)

    @attr.gpu
    def test_rdiv_forward_gpu(self):
        self.forward_gpu(lambda x, y: y / x)

    @attr.gpu
    def test_pow_forward_gpu(self):
        self.forward_gpu(lambda x, y: x ** y)

    @attr.gpu
    def test_rpow_forward_gpu(self):
        self.forward_gpu(lambda x, y: y ** x)

    def forward_chainerx(self, op, orig_xp):
        xs_chx = arrays_to_chainerx(orig_xp, (self.x,))
        self.check_forward(op, *xs_chx)

    @attr.chainerx
    def test_add_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: x + y, numpy)

    @attr.chainerx
    def test_radd_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: y + x, numpy)

    @attr.chainerx
    def test_sub_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: x - y, numpy)

    @attr.chainerx
    def test_rsub_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: y - x, numpy)

    @attr.chainerx
    def test_mul_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: x * y, numpy)

    @attr.chainerx
    def test_rmul_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: y * x, numpy)

    @attr.chainerx
    def test_div_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: x / y, numpy)

    @attr.chainerx
    def test_rdiv_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: y / x, numpy)

    @attr.chainerx
    def test_pow_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: x ** y, numpy)

    @attr.chainerx
    def test_rpow_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: y ** x, numpy)

    @attr.chainerx
    @attr.gpu
    def test_add_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: x + y, cuda.cupy)

    @attr.chainerx
    @attr.gpu
    def test_radd_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: y + x, cuda.cupy)

    @attr.chainerx
    @attr.gpu
    def test_sub_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: x - y, cuda.cupy)

    @attr.chainerx
    @attr.gpu
    def test_rsub_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: y - x, cuda.cupy)

    @attr.chainerx
    @attr.gpu
    def test_mul_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: x * y, cuda.cupy)

    @attr.chainerx
    @attr.gpu
    def test_rmul_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: y * x, cuda.cupy)

    @attr.chainerx
    @attr.gpu
    def test_div_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: x / y, cuda.cupy)

    @attr.chainerx
    @attr.gpu
    def test_rdiv_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: y / x, cuda.cupy)

    @attr.chainerx
    @attr.gpu
    def test_pow_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: x ** y, cuda.cupy)

    @attr.chainerx
    @attr.gpu
    def test_rpow_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: y ** x, cuda.cupy)

    def check_backward(self, op, x_data, y_grad):
        options = {}
        if self.dtype == numpy.float16:
            options = {'atol': 5e-3, 'rtol': 5e-2}
        gradient_check.check_backward(lambda x: op(x, self.value),
                                      x_data, y_grad,
                                      dtype=numpy.float64, **options)

    def backward_cpu(self, op):
        self.check_backward(op, self.x, self.gy)

    def test_add_backward_cpu(self):
        self.backward_cpu(lambda x, y: x + y)

    def test_radd_backward_cpu(self):
        self.backward_cpu(lambda x, y: y + x)

    def test_sub_backward_cpu(self):
        self.backward_cpu(lambda x, y: x - y)

    def test_rsub_backward_cpu(self):
        self.backward_cpu(lambda x, y: y - x)

    def test_mul_backward_cpu(self):
        self.backward_cpu(lambda x, y: x * y)

    def test_rmul_backward_cpu(self):
        self.backward_cpu(lambda x, y: y * x)

    def test_div_backward_cpu(self):
        self.backward_cpu(lambda x, y: x / y)

    def test_rdiv_backward_cpu(self):
        self.backward_cpu(lambda x, y: y / x)

    def test_pow_backward_cpu(self):
        self.backward_cpu(lambda x, y: x ** y)

    def test_rpow_backward_cpu(self):
        self.backward_cpu(lambda x, y: y ** x)

    def backward_gpu(self, op):
        self.check_backward(op, cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    def test_add_backward_gpu(self):
        self.backward_gpu(lambda x, y: x + y)

    @attr.gpu
    def test_radd_backward_gpu(self):
        self.backward_gpu(lambda x, y: y + x)

    @attr.gpu
    def test_sub_backward_gpu(self):
        self.backward_gpu(lambda x, y: x - y)

    @attr.gpu
    def test_rsub_backward_gpu(self):
        self.backward_gpu(lambda x, y: y - x)

    @attr.gpu
    def test_mul_backward_gpu(self):
        self.backward_gpu(lambda x, y: x * y)

    @attr.gpu
    def test_rmul_backward_gpu(self):
        self.backward_gpu(lambda x, y: y * x)

    @attr.gpu
    def test_div_backward_gpu(self):
        self.backward_gpu(lambda x, y: x / y)

    @attr.gpu
    def test_rdiv_backward_gpu(self):
        self.backward_gpu(lambda x, y: y / x)

    @attr.gpu
    def test_pow_backward_gpu(self):
        self.backward_gpu(lambda x, y: x ** y)

    @attr.gpu
    def test_rpow_backward_gpu(self):
        self.backward_gpu(lambda x, y: y ** x)

    def backward_chainerx(self, op):
        self.check_backward(
            op, chainerx.array(self.x), chainerx.array(self.gy))

    @attr.chainerx
    def test_add_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: x + y)

    @attr.chainerx
    def test_radd_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: y + x)

    @attr.chainerx
    def test_sub_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: x - y)

    @attr.chainerx
    def test_rsub_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: y - x)

    @attr.chainerx
    def test_mul_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: x * y)

    @attr.chainerx
    def test_rmul_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: y * x)

    @attr.chainerx
    def test_div_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: x / y)

    @attr.chainerx
    def test_rdiv_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: y / x)

    @attr.chainerx
    def test_pow_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: x ** y)

    @attr.chainerx
    def test_rpow_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: y ** x)

    def check_double_backward(self, op, x_data, y_grad, x_grad_grad):
        options = {}
        if self.dtype == numpy.float16:
            options = {'atol': 5e-3, 'rtol': 5e-2}

        def _op(x):
            return op(x, self.value)

        gradient_check.check_double_backward(
            _op, x_data, y_grad, x_grad_grad, dtype=numpy.float64, **options)

    def double_backward_cpu(self, op):
        self.check_double_backward(op, self.x, self.gy, self.ggx)

    def test_pow_double_backward_cpu(self):
        self.double_backward_cpu(lambda x, y: x ** y)

    def test_rpow_double_backward_cpu(self):
        self.double_backward_cpu(lambda x, y: y ** x)

    def test_rdiv_double_backward_cpu(self):
        self.double_backward_cpu(lambda x, y: y / x)

    def double_backward_gpu(self, op):
        self.check_double_backward(
            op, cuda.to_gpu(self.x), cuda.to_gpu(self.gy),
            cuda.to_gpu(self.ggx))

    @attr.gpu
    def test_pow_double_backward_gpu(self):
        self.double_backward_gpu(lambda x, y: x ** y)

    @attr.gpu
    def test_rpow_double_backward_gpu(self):
        self.double_backward_gpu(lambda x, y: y ** x)

    @attr.gpu
    def test_rdiv_double_backward_gpu(self):
        self.double_backward_gpu(lambda x, y: y / x)

    def double_backward_chainerx(self, op):
        self.check_double_backward(
            op, chainerx.array(self.x), chainerx.array(self.gy),
            chainerx.array(self.ggx))

    @attr.chainerx
    def test_pow_double_backward_chainerx(self):
        # TODO(niboshi): Support it
        raise unittest.SkipTest('chainerx.broadcast is required')

        self.double_backward_chainerx(lambda x, y: x ** y)

    @attr.chainerx
    def test_rpow_double_backward_chainerx(self):
        # TODO(niboshi): Support it
        raise unittest.SkipTest(
            'chainerx.log with scalar argument is required')

        self.double_backward_chainerx(lambda x, y: y ** x)

    @attr.chainerx
    def test_rdiv_double_backward_chainerx(self):
        # TODO(niboshi): Support it
        raise unittest.SkipTest('chainerx.broadcast is required')

        self.double_backward_chainerx(lambda x, y: y / x)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestVariableConstantArrayOp(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(.5, 1, (3, 2)).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, (3, 2)).astype(self.dtype)
        self.ggx = numpy.random.uniform(.5, 1, (3, 2)).astype(self.dtype)
        self.value = numpy.random.uniform(-1, 1, (3, 2)).astype(self.dtype)

    def check_forward(self, op, array_conv, positive):
        value = self.value
        if positive:
            value = numpy.abs(value)
        v = array_conv(value)
        x = chainer.Variable(array_conv(self.x))
        y = op(x, v)
        if self.dtype == numpy.float16:
            tol = 1e-3
        else:
            tol = 1e-6

        testing.assert_allclose(
            op(self.x, value), y.data, atol=tol, rtol=tol)

    def forward_cpu(self, op, positive=False):
        self.check_forward(op, lambda x: x, positive)

    def test_add_forward_cpu(self):
        self.forward_cpu(lambda x, y: x + y)

    def test_radd_forward_cpu(self):
        self.forward_cpu(lambda x, y: y + x)

    def test_sub_forward_cpu(self):
        self.forward_cpu(lambda x, y: x - y)

    def test_rsub_forward_cpu(self):
        self.forward_cpu(lambda x, y: y - x)

    def test_mul_forward_cpu(self):
        self.forward_cpu(lambda x, y: x * y)

    def test_rmul_forward_cpu(self):
        self.forward_cpu(lambda x, y: y * x)

    def test_div_forward_cpu(self):
        self.forward_cpu(lambda x, y: x / y)

    def test_rdiv_forward_cpu(self):
        self.forward_cpu(lambda x, y: y / x)

    def test_pow_forward_cpu(self):
        self.forward_cpu(lambda x, y: x ** y)

    def test_rpow_forward_cpu(self):
        self.forward_cpu(lambda x, y: y ** x, positive=True)

    def forward_gpu(self, op, positive=False):
        self.check_forward(op, cuda.to_gpu, positive)

    @attr.gpu
    def test_add_forward_gpu(self):
        self.forward_gpu(lambda x, y: x + y)

    @attr.gpu
    def test_radd_forward_gpu(self):
        self.forward_gpu(lambda x, y: y + x)

    @attr.gpu
    def test_sub_forward_gpu(self):
        self.forward_gpu(lambda x, y: x - y)

    @attr.gpu
    def test_rsub_forward_gpu(self):
        self.forward_gpu(lambda x, y: y - x)

    @attr.gpu
    def test_mul_forward_gpu(self):
        self.forward_gpu(lambda x, y: x * y)

    @attr.gpu
    def test_rmul_forward_gpu(self):
        self.forward_gpu(lambda x, y: y * x)

    @attr.gpu
    def test_div_forward_gpu(self):
        self.forward_gpu(lambda x, y: x / y)

    @attr.gpu
    def test_rdiv_forward_gpu(self):
        self.forward_gpu(lambda x, y: y / x)

    @attr.gpu
    def test_pow_forward_gpu(self):
        self.forward_gpu(lambda x, y: x ** y)

    @attr.gpu
    def test_rpow_forward_gpu(self):
        self.forward_gpu(lambda x, y: y ** x, positive=True)

    def forward_chainerx(self, op, orig_xp, positive=False):
        if orig_xp is numpy:
            array_conv = chainer.backend.to_chainerx
        else:
            assert orig_xp is cuda.cupy

            def array_conv(x):
                return chainer.backend.to_chainerx(cuda.to_gpu(x))
        self.check_forward(op, array_conv, positive)

    @attr.chainerx
    def test_pow_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: x ** y, numpy)

    @attr.chainerx
    def test_rpow_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x, y: y ** x, numpy, positive=True)

    @attr.chainerx
    @attr.gpu
    def test_pow_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: x ** y, cuda.cupy)

    @attr.chainerx
    @attr.gpu
    def test_rpow_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x, y: y ** x, cuda.cupy, positive=True)

    def check_backward(self, op, x_data, y_grad, array_conv, positive):
        value = self.value
        if positive:
            value = numpy.abs(value)
        value = array_conv(value)
        x_data = array_conv(x_data)
        y_grad = array_conv(y_grad)
        options = {}
        if self.dtype == numpy.float16:
            options = {'atol': 5e-3, 'rtol': 5e-2}

        # numeric_gradient will cast `x` to float64, but not `value`.
        # It's casted here.
        def op_(x):
            return op(
                x,
                value if x.dtype == value.dtype else value.astype(x.dtype))

        gradient_check.check_backward(op_, x_data, y_grad,
                                      dtype=numpy.float64, **options)

    def backward_cpu(self, op, positive=False):
        self.check_backward(op, self.x, self.gy, lambda x: x, positive)

    def test_add_backward_cpu(self):
        self.backward_cpu(lambda x, y: x + y)

    def test_radd_backward_cpu(self):
        self.backward_cpu(lambda x, y: y + x)

    def test_sub_backward_cpu(self):
        self.backward_cpu(lambda x, y: x - y)

    def test_rsub_backward_cpu(self):
        self.backward_cpu(lambda x, y: y - x)

    def test_mul_backward_cpu(self):
        self.backward_cpu(lambda x, y: x * y)

    def test_rmul_backward_cpu(self):
        self.backward_cpu(lambda x, y: y * x)

    def test_div_backward_cpu(self):
        self.backward_cpu(lambda x, y: x / y)

    def test_rdiv_backward_cpu(self):
        self.backward_cpu(lambda x, y: y / x)

    def test_pow_backward_cpu(self):
        self.backward_cpu(lambda x, y: x ** y)

    def test_rpow_backward_cpu(self):
        self.backward_cpu(lambda x, y: y ** x, positive=True)

    def backward_gpu(self, op, positive=False):
        self.check_backward(op, self.x, self.gy, cuda.to_gpu, positive)

    @attr.gpu
    def test_add_backward_gpu(self):
        self.backward_gpu(lambda x, y: x + y)

    @attr.gpu
    def test_radd_backward_gpu(self):
        self.backward_gpu(lambda x, y: y + x)

    @attr.gpu
    def test_sub_backward_gpu(self):
        self.backward_gpu(lambda x, y: x - y)

    @attr.gpu
    def test_mul_backward_gpu(self):
        self.backward_gpu(lambda x, y: x * y)

    @attr.gpu
    def test_rmul_backward_gpu(self):
        self.backward_gpu(lambda x, y: y * x)

    @attr.gpu
    def test_div_backward_gpu(self):
        self.backward_gpu(lambda x, y: x / y)

    @attr.gpu
    def test_rdiv_backward_gpu(self):
        self.backward_gpu(lambda x, y: y / x)

    @attr.gpu
    def test_pow_backward_gpu(self):
        self.backward_gpu(lambda x, y: x ** y)

    @attr.gpu
    def test_rpow_backward_gpu(self):
        self.backward_gpu(lambda x, y: y ** x, positive=True)

    def backward_chainerx(self, op, positive=False):
        self.check_backward(op, self.x, self.gy, chainerx.array, positive)

    @attr.chainerx
    def test_add_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: x + y)

    @attr.chainerx
    def test_radd_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: y + x)

    @attr.chainerx
    def test_sub_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: x - y)

    @attr.chainerx
    def test_mul_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: x * y)

    @attr.chainerx
    def test_rmul_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: y * x)

    @attr.chainerx
    def test_div_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: x / y)

    @attr.chainerx
    def test_rdiv_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: y / x)

    @attr.chainerx
    def test_pow_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: x ** y)

    @attr.chainerx
    def test_rpow_backward_chainerx(self):
        self.backward_chainerx(lambda x, y: y ** x, positive=True)

    def check_double_backward(
            self, op, x_data, y_grad, x_grad_grad, array_conv, positive):
        value = self.value
        if positive:
            value = numpy.abs(value)
        value = array_conv(value)
        x_data = array_conv(x_data)
        y_grad = array_conv(y_grad)
        x_grad_grad = array_conv(x_grad_grad)
        options = {}
        if self.dtype == numpy.float16:
            options = {'atol': 5e-3, 'rtol': 5e-2}

        def _op(x):
            return op(x, value)

        gradient_check.check_double_backward(
            _op, x_data, y_grad, x_grad_grad, dtype=numpy.float64, **options)

    def double_backward_cpu(self, op, positive=False):
        self.check_double_backward(
            op, self.x, self.gy, self.ggx, lambda x: x, positive)

    def test_pow_double_backward_cpu(self):
        self.double_backward_cpu(lambda x, y: x ** y)

    def test_rpow_double_backward_cpu(self):
        self.double_backward_cpu(lambda x, y: y ** x, positive=True)

    def double_backward_gpu(self, op, positive=False):
        self.check_double_backward(
            op, self.x, self.gy, self.ggx, cuda.to_gpu, positive)

    @attr.gpu
    def test_pow_double_backward_gpu(self):
        self.double_backward_gpu(lambda x, y: x ** y)

    @attr.gpu
    def test_rpow_double_backward_gpu(self):
        self.double_backward_gpu(lambda x, y: y ** x, positive=True)

    def double_backward_chainerx(self, op, positive=False):
        self.check_double_backward(
            op, self.x, self.gy, self.ggx, chainerx.array, positive)

    @attr.chainerx
    def test_pow_double_backward_chainerx(self):
        self.double_backward_chainerx(lambda x, y: x ** y)

    @attr.chainerx
    def test_rpow_double_backward_chainerx(self):
        self.double_backward_chainerx(lambda x, y: y ** x, positive=True)


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestUnaryFunctions(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        for i in numpy.ndindex(self.shape):
            if -0.1 < self.x[i] < 0.1:
                self.x[i] = 0.5
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(.5, 1, self.shape).astype(self.dtype)

    def check_forward(self, op, op_np, x_data):
        x = chainer.Variable(x_data)
        y = op(x)
        testing.assert_allclose(
            op_np(self.x), y.data, atol=1e-7, rtol=1e-7)

    def forward_cpu(self, op, op_np):
        self.check_forward(op, op_np, self.x)

    def test_neg_forward_cpu(self):
        self.forward_cpu(lambda x: -x, lambda x: -x)

    def test_abs_forward_cpu(self):
        self.forward_cpu(lambda x: abs(x), lambda x: abs(x))

    def forward_gpu(self, op, op_np):
        self.check_forward(op, op_np, cuda.to_gpu(self.x))

    @attr.gpu
    def test_neg_forward_gpu(self):
        self.forward_gpu(lambda x: -x, lambda x: -x)

    @attr.gpu
    def test_abs_forward_gpu(self):
        self.forward_gpu(lambda x: abs(x), lambda x: abs(x))

    def forward_chainerx(self, op, op_np, orig_xp):
        xs_chx = arrays_to_chainerx(orig_xp, (self.x,))
        self.check_forward(op, op_np, *xs_chx)

    @attr.chainerx
    def test_neg_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x: -x, lambda x: -x, numpy)

    @attr.chainerx
    def test_abs_forward_chainerx_cpu(self):
        self.forward_chainerx(lambda x: abs(x), lambda x: abs(x), numpy)

    @attr.chainerx
    @attr.gpu
    def test_neg_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x: -x, lambda x: -x, cuda.cupy)

    @attr.chainerx
    @attr.gpu
    def test_abs_forward_chainerx_gpu(self):
        self.forward_chainerx(lambda x: abs(x), lambda x: abs(x), cuda.cupy)

    def check_backward(self, op, x_data, y_grad):
        options = {}
        if self.dtype == numpy.float16:
            options = {'atol': 5e-3, 'rtol': 5e-2}
        gradient_check.check_backward(
            op, x_data, y_grad, dtype=numpy.float64, **options)

    def backward_cpu(self, op):
        self.check_backward(op, self.x, self.gy)

    def test_neg_backward_cpu(self):
        self.backward_cpu(lambda x: -x)

    def test_abs_backward_cpu(self):
        self.backward_cpu(lambda x: abs(x))

    def backward_gpu(self, op):
        self.check_backward(op, cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    @attr.gpu
    def test_neg_backward_gpu(self):
        self.backward_gpu(lambda x: -x)

    @attr.gpu
    def test_abs_backward_gpu(self):
        self.backward_gpu(lambda x: abs(x))

    def backward_chainerx(self, op):
        self.check_backward(
            op, chainerx.array(self.x), chainerx.array(self.gy))

    @attr.chainerx
    def test_neg_backward_chainerx(self):
        self.backward_chainerx(lambda x: -x)

    def check_double_backward(self, op, x_data, y_grad, x_grad_grad):
        options = {}
        if self.dtype == numpy.float16:
            options = {'atol': 5e-3, 'rtol': 5e-2}

        gradient_check.check_double_backward(
            op, x_data, y_grad, x_grad_grad, dtype=numpy.float64, **options)

    def double_backward_cpu(self, op):
        self.check_double_backward(op, self.x, self.gy, self.ggx)

    def test_neg_double_backward_cpu(self):
        self.double_backward_cpu(lambda x: -x)

    def test_abs_double_backward_cpu(self):
        self.double_backward_cpu(lambda x: abs(x))

    def double_backward_gpu(self, op):
        self.check_double_backward(
            op, cuda.to_gpu(self.x), cuda.to_gpu(self.gy),
            cuda.to_gpu(self.ggx))

    @attr.gpu
    def test_neg_double_backward_gpu(self):
        self.double_backward_gpu(lambda x: -x)

    @attr.gpu
    def test_abs_double_backward_gpu(self):
        self.double_backward_gpu(lambda x: abs(x))

    def double_backward_chainerx(self, op):
        self.check_double_backward(
            op, chainerx.array(self.x), chainerx.array(self.gy),
            chainerx.array(self.ggx))

    @attr.chainerx
    def test_neg_double_backward_chainerx(self):
        self.double_backward_chainerx(lambda x: -x)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestNegativePow(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 0, (3, 2)).astype(self.dtype)
        self.gy = numpy.random.uniform(-1, 1, (3, 2)).astype(self.dtype)
        self.ggx = numpy.random.uniform(-1, 1, (3, 2)).astype(self.dtype)

    def check_backward(self, x_data, y_grad):
        options = {}
        if self.dtype == numpy.float16:
            options = {'atol': 5e-3, 'rtol': 5e-2}
        gradient_check.check_backward(
            lambda x: x ** 2, x_data, y_grad, dtype=numpy.float64, **options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def check_double_backward(self, x_data, y_grad, x_grad_grad):
        options = {}
        if self.dtype == numpy.float16:
            options = {'atol': 5e-3, 'rtol': 5e-2}
        gradient_check.check_double_backward(
            lambda x: x ** 2, x_data, y_grad, x_grad_grad, dtype=numpy.float64,
            **options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggx))


@testing.parameterize(*testing.product_dict(
    [
        {'left_const': False, 'right_const': False},
        {'left_const': True, 'right_const': False},
        {'left_const': False, 'right_const': True},
    ], [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ], [
        {'x_shape': (3, 2), 'y_shape': (2, 4), 'z_shape': (3, 4)},
        {'x_shape': (2, 3, 2), 'y_shape': (2, 2, 4), 'z_shape': (2, 3, 4)},
        {'x_shape': (2, 1, 3, 4),
         'y_shape': (2, 4, 2),
         'z_shape': (2, 2, 3, 2)},
        {'x_shape': (5, 3, 2), 'y_shape': (2,), 'z_shape': (5, 3)},
        {'x_shape': (2,), 'y_shape': (5, 2, 4), 'z_shape': (5, 4)},
        {'x_shape': (2, 3, 2), 'y_shape': (2, 4), 'z_shape': (2, 3, 4)},
        {'x_shape': (3,), 'y_shape': (3,), 'z_shape': ()},
    ]
))
@unittest.skipUnless(sys.version_info >= (3, 5),
                     'Only for Python3.5 or higher')
class TestMatMul(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        self.y = numpy.random.uniform(-1, 1, self.y_shape).astype(self.dtype)
        self.gz = numpy.random.uniform(-1, 1, self.z_shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(
            -1, 1, self.x_shape).astype(self.dtype)
        self.ggy = numpy.random.uniform(
            -1, 1, self.y_shape).astype(self.dtype)

    def _get_forward_answer(self, x, y):
        if x.ndim <= 2 or y.ndim == 1:
            return numpy.dot(x, y)
        elif hasattr(numpy, 'matmul'):
            # Note: NumPy 1.14.0 has a bug in einsum (numpy/numpy#10343),
            # so we use matmul if available to avoid it
            return numpy.matmul(x, y)
        else:
            return numpy.einsum('...ij,...jk->...ik', x, y)

    def check_forward(self, x_data, y_data):
        if self.left_const:
            x = x_data
        else:
            x = chainer.Variable(x_data)
        if self.right_const:
            y = y_data
        else:
            y = chainer.Variable(y_data)
        z = operator.matmul(x, y)
        if self.dtype == numpy.float16:
            options = {'atol': 2e-3, 'rtol': 2e-3}
        else:
            options = {'atol': 2e-7, 'rtol': 2e-7}
        testing.assert_allclose(
            self._get_forward_answer(self.x, self.y), z.data, **options)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.y)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.y))

    def check_backward(self, x_data, y_data, z_grad):
        if self.right_const:
            def op(x):
                return operator.matmul(x, y_data)
            data = x_data,
        elif self.left_const:
            def op(y):
                return operator.matmul(x_data, y)
            data = y_data,
        else:
            op = operator.matmul
            data = x_data, y_data

        if self.dtype == numpy.float16:
            options = {'atol': 1e-3, 'rtol': 1e-2}
        else:
            options = {'atol': 1e-4, 'rtol': 1e-4}
        gradient_check.check_backward(
            op, data, z_grad, dtype=numpy.float64, **options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.y, self.gz)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.y), cuda.to_gpu(self.gz))

    def check_double_backward(
            self, x_data, y_data, z_grad, x_grad_grad, y_grad_grad):
        if self.right_const:
            def op(x):
                return operator.matmul(x, y_data.astype(x.dtype))
            data = x_data,
            grad_grad = x_grad_grad,
        elif self.left_const:
            def op(y):
                return operator.matmul(x_data.astype(y.dtype), y)
            data = y_data,
            grad_grad = y_grad_grad,
        else:
            op = operator.matmul
            data = x_data, y_data
            grad_grad = x_grad_grad, y_grad_grad

        if self.dtype == numpy.float16:
            options = {'atol': 1e-3, 'rtol': 1e-2}
        else:
            options = {'atol': 1e-4, 'rtol': 1e-4}
        gradient_check.check_double_backward(
            op, data, z_grad, grad_grad, dtype=numpy.float64, **options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.y, self.gz, self.ggx, self.ggy)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.y), cuda.to_gpu(self.gz),
            cuda.to_gpu(self.ggx), cuda.to_gpu(self.ggy))


@testing.parameterize(
    {'x_shape': (), 'y_shape': ()},
    {'x_shape': (3, 2), 'y_shape': ()},
    {'x_shape': (), 'y_shape': (2, 4)},
    {'x_shape': (2, 3), 'y_shape': (2, 3)},
    {'x_shape': (2,), 'y_shape': (1,)},
)
@unittest.skipUnless(sys.version_info >= (3, 5),
                     'Only for Python3.5 or higher')
class TestMatMulInvalidShape(unittest.TestCase):

    dtype = numpy.float32

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.x_shape).astype(self.dtype)
        self.y = numpy.random.uniform(-1, 1, self.y_shape).astype(self.dtype)

    def test_invalid_type(self):
        x = chainer.Variable(self.x)
        y = chainer.Variable(self.y)
        with pytest.raises(type_check.InvalidType):
            operator.matmul(x, y)


class TestNotSupportOperation(unittest.TestCase):

    def setUp(self):
        self.x = chainer.Variable(numpy.zeros(10))
        self.y = chainer.Variable(numpy.zeros(10))

    def test_lt(self):
        with pytest.raises(NotImplementedError):
            self.x < self.y

    def test_le(self):
        with pytest.raises(NotImplementedError):
            self.x <= self.y

    def test_eq(self):
        with pytest.raises(NotImplementedError):
            self.x == self.y

    def test_ne(self):
        with pytest.raises(NotImplementedError):
            self.x != self.y

    def test_gt(self):
        with pytest.raises(NotImplementedError):
            self.x > self.y

    def test_ge(self):
        with pytest.raises(NotImplementedError):
            self.x >= self.y

    def test_nonzero(self):
        with pytest.raises(NotImplementedError):
            if self.x:
                pass


class ConvertValueToStringTest(unittest.TestCase):

    def _check_scalar(self, value, string):
        self.assertEqual(basic_math._convert_value_to_string(value), string)

    def test_integer_positive(self):
        self._check_scalar(2, '2')

    def test_integer_zero(self):
        self._check_scalar(0, '0')

    def test_integer_negative(self):
        self._check_scalar(-2, '(-2)')

    def test_float_positive(self):
        self._check_scalar(2.0, '2.0')

    def test_float_zero(self):
        self._check_scalar(0.0, '0.0')

    def test_float_negative(self):
        self._check_scalar(-2.0, '(-2.0)')

    def test_numpy_scalar(self):
        self._check_scalar(numpy.float32(2), '2.0')

    def _check_array(self, value, string):
        self.assertEqual(basic_math._convert_value_to_string(value), string)
        value = chainer.Variable(value)
        self.assertEqual(basic_math._convert_value_to_string(value), string)

    def test_array_cpu(self):
        self._check_array(numpy.array([1, 2]), 'constant array')

    @attr.gpu
    def test_array_gpu(self):
        self._check_array(cuda.ndarray([1, 2]), 'constant array')


class TestLabel(unittest.TestCase):

    def test_neg(self):
        self.assertEqual(basic_math.Neg().label, '__neg__')

    def test_absolute(self):
        self.assertEqual(basic_math.Absolute().label, '|_|')

    def test_add(self):
        self.assertEqual(basic_math.Add().label, '_ + _')

    def test_add_constant(self):
        self.assertEqual(basic_math.AddConstant(2.0).label, '_ + 2.0')

    def test_sub(self):
        self.assertEqual(basic_math.Sub().label, '_ - _')

    def test_sub_from_constant(self):
        self.assertEqual(basic_math.SubFromConstant(2.0).label, '2.0 - _')

    def test_mul(self):
        self.assertEqual(basic_math.Mul().label, '_ * _')

    def test_mul_constant(self):
        self.assertEqual(basic_math.MulConstant(2.0).label, '_ * 2.0')

    def test_div(self):
        self.assertEqual(basic_math.Div().label, '_ / _')

    def test_div_from_constant(self):
        self.assertEqual(basic_math.DivFromConstant(2.0).label, '2.0 / _')

    def test_pow_var_var(self):
        self.assertEqual(basic_math.PowVarVar().label, '_ ** _')

    def test_pow_var_const(self):
        self.assertEqual(basic_math.PowVarConst(2.0).label, '_ ** 2.0')

    def test_pow_const_var(self):
        self.assertEqual(basic_math.PowConstVar(2.0).label, '2.0 ** _')


testing.run_module(__name__, __file__)
