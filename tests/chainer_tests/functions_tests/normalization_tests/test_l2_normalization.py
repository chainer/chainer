import functools
import unittest

import itertools
import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


def _skip_if(cond, reason):
    """Skip test if cond(self) is True"""
    def decorator(impl):
        @functools.wraps(impl)
        def wrapper(self, *args, **kwargs):
            if cond(self):
                raise unittest.SkipTest(reason)
            else:
                impl(self, *args, **kwargs)
        return wrapper
    return decorator


def _is_good_param(param):
    # Check if 'nonzero' param is valid and meaningful. On the latter point,
    # x should contain at least a zero if 'nonzeros' param is given.
    return param['nonzeros'] is None \
        or param['nonzeros'] < numpy.prod(param['shape'])


@testing.parameterize(*filter(_is_good_param, testing.product([
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
    [
        {'shape': (4, 15), 'axis': 1},
        {'shape': (4,), 'axis': 0},
        {'shape': (4, 3, 2, 5), 'axis': 0},
        {'shape': (4, 3, 2, 5), 'axis': 1},
        {'shape': (4, 3, 2, 5), 'axis': 2},
        {'shape': (4, 3, 2, 5), 'axis': 3},
        {'shape': (4, 3, 2), 'axis': (0, 1)},
        {'shape': (4, 3, 2, 4, 3, 2, 2), 'axis': (1, 4, 3, 6)},
        {'shape': (0, 2), 'axis': 1},
        {'shape': (), 'axis': ()},
    ],
    [
        # nonzeros (optional int): number of nonzero elems in input
        # truezero (bool): flag whether zero elems are exactly zero. If false,
        #     randomly-chosen small values are used.
        {'eps': 1e-5, 'nonzeros': None},
        {'eps': 1e-1, 'nonzeros': None},
        {'eps': 1e-1, 'nonzeros': 0, 'truezero': True},
        {'eps': 1e-1, 'nonzeros': 0, 'truezero': False},
        {'eps': 1e-1, 'nonzeros': 2, 'truezero': True},
        {'eps': 1e-1, 'nonzeros': 2, 'truezero': False},
    ],
])))
class TestL2Normalization(unittest.TestCase):

    def setUp(self):
        min_abs = 0.1
        if self.dtype == numpy.float16:
            tuple_axis = self.axis
            if not isinstance(tuple_axis, tuple):
                tuple_axis = (tuple_axis,)
            aggr_size = numpy.prod([self.shape[i] for i in tuple_axis], dtype=int)
            min_abs = max(min_abs, 0.5 / aggr_size)
        self.x = chainer.utils.force_array(
            numpy.random.uniform(min_abs, 1, self.shape)
            * (1 - 2 * numpy.random.randint(2, size=self.shape)),
            self.dtype)
        if self.nonzeros is not None:
            # Make self.x have limited number of large values

            # get mask of indices to modify at
            zeros = self.x.size - self.nonzeros
            while True:
                rand = numpy.random.uniform(0, 1, self.shape)
                mask = rand <= numpy.sort(rand.ravel())[zeros - 1]
                if self.x[mask].shape == (zeros,):
                    break

            # set zeros or small values to a part of the input
            if self.truezero:
                self.x[mask] = 0
            else:
                zero_scale = 10. ** numpy.random.randint(-40, -3)
                self.x[mask] = numpy.random.uniform(
                    -zero_scale, zero_scale, zeros)
        self.gy = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.ggx = numpy.random.uniform(
            -1, 1, self.shape).astype(self.dtype)

        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-3}
        else:
            self.check_forward_options = {}

        if self.nonzeros is None:
            if self.dtype == numpy.float16:
                self.check_backward_options = {
                    'dtype': numpy.float64, 'atol': 5e-3, 'rtol': 5e-3}
                self.check_double_backward_options = {
                    'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2}
            else:
                self.check_backward_options = {
                    'dtype': numpy.float64, 'atol': 1e-4, 'rtol': 1e-4}
                self.check_double_backward_options = {
                    'dtype': numpy.float64, 'atol': 1e-4, 'rtol': 1e-4}
        else:
            self.check_backward_options = {
                'dtype': numpy.float64, 'atol': 1e-2, 'rtol': 1e-2,
                'eps': 1e-4}

    def check_forward(self, x_data, axis):
        eps = self.eps
        x = chainer.Variable(x_data)

        y = functions.normalize(x, eps=eps, axis=axis)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)

        y_expect = numpy.empty_like(self.x)
        shape = self.x.shape
        indices = []
        axis_tuple = axis if isinstance(axis, tuple) else (axis,)
        for i in six.moves.range(len(shape)):
            if i not in axis_tuple:
                indices.append(six.moves.range(shape[i]))
            else:
                indices.append([slice(None)])
        indices_tuple = list(itertools.product(*indices))
        for index in indices_tuple:
            # Note: Casting back the result of `numpy.linalg.norm` to `x.dtype`
            # because old NumPy casts it to float32 when a float16 value is
            # given.
            numerator = numpy.linalg.norm(self.x[index]).astype(x.dtype) + eps
            y_expect[index] = self.x[index] / numerator
        testing.assert_allclose(y_expect, y_data, **self.check_forward_options)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.axis)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), self.axis)

    def check_backward(self, x_data, axis, y_grad):
        def f(x):
            return functions.normalize(x, eps=self.eps, axis=axis)

        gradient_check.check_backward(
            f, x_data, y_grad, **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.axis, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), self.axis, cuda.to_gpu(self.gy))

    @_skip_if(
        lambda self: self.nonzeros is not None,
        'backward of L2Normalize is non-differentiable at zero vector')
    def check_double_backward(self, x_data, axis, y_grad, x_grad_grad):
        def f(x):
            return functions.normalize(x, eps=self.eps, axis=axis)

        gradient_check.check_double_backward(
            f, x_data, y_grad, x_grad_grad,
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.axis, self.gy, self.ggx)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(
            cuda.to_gpu(self.x), self.axis, cuda.to_gpu(self.gy),
            cuda.to_gpu(self.ggx))

    def check_eps(self, x_data):
        x = chainer.Variable(x_data)

        y = functions.normalize(x, axis=self.axis)
        self.assertEqual(y.data.dtype, self.dtype)
        y_data = cuda.to_cpu(y.data)

        y_expect = numpy.zeros_like(self.x)
        testing.assert_allclose(y_expect, y_data)

    def test_eps_cpu(self):
        self.check_eps(numpy.zeros_like(self.x))

    @attr.gpu
    def test_eps_gpu(self):
        self.check_eps(cuda.to_gpu(numpy.zeros_like(self.x)))


testing.run_module(__name__, __file__)
