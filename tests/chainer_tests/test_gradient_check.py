import math
import unittest
import warnings

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import backend
from chainer.testing import condition
import chainerx


def _uniform(*shape):
    return numpy.random.uniform(-1, 1, shape).astype(numpy.float32)


def _full_like(x, val):
    xp = chainer.backend.get_array_module(x)
    return xp.full_like(x, val)


def _zeros_like(x):
    xp = chainer.backend.get_array_module(x)
    return xp.zeros_like(x)


def _dot(x, y):
    return sum(map(lambda a: a[0] * a[1], zip(x, y)))


class NumericalGradientTest(unittest.TestCase):

    in_shapes = ((2, 1),)
    gout_shapes = ((2, 1),)
    eps = None
    atol = 1e-3
    rtol = 1e-3

    def f(self, xs):
        return xs[0] ** 2,

    def df(self, xs):
        return (2 * xs[0],),

    def setUp(self):
        self.xs = tuple([_uniform(*s) for s in self.in_shapes])
        self.gys = tuple([
            None if s is None else _uniform(*s) for s in self.gout_shapes])

    def check_numerical_grad_one(self, f, df, xs, gys, eps):
        dfxs = df(xs)

        gys = tuple(0 if gy is None else gy for gy in gys)
        # matrix-vector multiplication of dfxs and dys
        dx_expect = tuple(map(lambda dfx: _dot(dfx, gys), dfxs))

        def func():
            return f(xs)
        dx_actual = gradient_check.numerical_grad(func, xs, gys, eps)

        self.assertEqual(len(dx_expect), len(dx_actual))
        for e, a in zip(dx_expect, dx_actual):
            testing.assert_allclose(e, a, atol=self.atol, rtol=self.rtol)

    def check_numerical_grad(self, f, df, xs, gys, eps=None):
        if eps is None:
            eps = tuple(10 ** (-i) for i in six.moves.range(2, 5))
        elif not isinstance(eps, tuple):
            eps = (eps, )

        for e in eps:
            self.check_numerical_grad_one(f, df, xs, gys, e)

    def test_numerical_grad_cpu(self):
        self.check_numerical_grad(self.f, self.df, self.xs, self.gys,
                                  eps=self.eps)

    @attr.gpu
    def test_numerical_grad_gpu(self):
        gys = tuple(None if gy is None else cuda.to_gpu(gy)
                    for gy in self.gys)

        self.check_numerical_grad(self.f, self.df,
                                  tuple(map(cuda.to_gpu, self.xs)), gys,
                                  eps=self.eps)


class NumericalGradientTest2(NumericalGradientTest):

    in_shapes = ((),)
    gout_shapes = ((),)

    def f(self, xs):
        return 1,

    def df(self, xs):
        return (0,),


class NumericalGradientTest3(NumericalGradientTest):

    in_shapes = ((2, 1),)
    gout_shapes = ((2, 1),)
    # Too small eps causes cancellation of significant digits
    eps = (1e-2, 1e-3)

    def f(self, xs):
        xp = chainer.backend.get_array_module(*xs)
        return xp.exp(xs[0]),

    def df(self, xs):
        xp = chainer.backend.get_array_module(*xs)
        return (xp.exp(xs[0]),),

    def setUp(self):
        self.xs = (_uniform(2, 1),)
        self.gys = (_uniform(2, 1),)


class NumericalGradientTest4(NumericalGradientTest):

    in_shapes = ((2, 1), (2, 1))
    gout_shapes = ((2, 1), (2, 1), (2, 1))
    atol = 1e-2
    rtol = 1e-2

    def f(self, xs):
        assert len(xs) == 2
        return (2 * xs[0] + 3 * xs[1],
                4 * xs[0] + 5 * xs[1],
                6 * xs[0] + 7 * xs[1])

    def df(self, xs):
        assert len(xs) == 2
        return (
            (_full_like(xs[0], 2), _full_like(xs[0], 4), _full_like(xs[0], 6)),
            (_full_like(xs[1], 3), _full_like(xs[1], 5), _full_like(xs[1], 7)))


class NumericalGradientTest5(NumericalGradientTest):

    in_shapes = ((2, 1), (2, 1))
    gout_shapes = ((2, 1), None, (2, 1))
    atol = 5e-3
    rtol = 5e-3

    def f(self, xs):
        assert len(xs) == 2
        return (2 * xs[0] + 3 * xs[1],
                4 * xs[0] + 5 * xs[1],
                6 * xs[0] + 7 * xs[1])

    def df(self, xs):
        assert len(xs) == 2
        return (
            (_full_like(xs[0], 2), _zeros_like(xs[0]), _full_like(xs[0], 6)),
            (_full_like(xs[1], 3), _zeros_like(xs[1]), _full_like(xs[1], 7)))


class NumericalGradientTest6(NumericalGradientTest):

    in_shapes = ((2, 1),)
    gout_shapes = (None,)


class NumericalGradientReferenceTest(unittest.TestCase):

    def setUp(self):
        self.x = _uniform(2, 3)

    def check_reference(self, x):
        # A returned value and an input refers the same memory.
        # See issue #488
        def func():
            return x,
        gx, = gradient_check.numerical_grad(func, (x,), (1,))
        testing.assert_allclose(cuda.to_cpu(gx), 1)

    def test_reference_cpu(self):
        self.check_reference(self.x)

    @attr.gpu
    def test_reference_gpu(self):
        self.check_reference(cuda.to_gpu(self.x))


class NumericalGradientInvalidEps(NumericalGradientTest):

    def check_invalid_eps(self, xs, gys, eps):
        with self.assertRaises(AssertionError):
            self.check_numerical_grad(self.f, self.df, xs, gys, eps)

    @condition.retry(3)
    def test_numerical_grad_cpu(self):
        self.check_invalid_eps(self.xs, self.gys, 0)
        self.check_invalid_eps(self.xs, self.gys, -1.0)

    @condition.retry(3)
    @attr.gpu
    def test_numerical_grad_gpu(self):
        xs = tuple(map(cuda.to_gpu, self.xs))
        gys = tuple(None if gy is None else cuda.to_gpu(gy)
                    for gy in self.gys)

        self.check_invalid_eps(xs, gys, 0)
        self.check_invalid_eps(xs, gys, -1.0)


class NumericalGradientInvalidType(unittest.TestCase):

    def setUp(self):
        self.x = numpy.array(0)
        self.y = numpy.array(0)
        self.f = lambda: None

    @attr.gpu
    def test_invalid_inputs(self):
        y = cuda.to_gpu(self.y)
        with self.assertRaises(RuntimeError):
            gradient_check.numerical_grad(self.f, (self.x, y), ())

    @attr.gpu
    def test_invalid_outputs(self):
        y = cuda.to_gpu(self.y)
        with self.assertRaises(RuntimeError):
            gradient_check.numerical_grad(self.f, (), (self.x, y))

    @attr.gpu
    def test_invalid_mixed(self):
        y = cuda.to_gpu(self.y)
        with self.assertRaises(RuntimeError):
            gradient_check.numerical_grad(self.f, (self.x,), (y,))


class NumericalGradientEpsTest(unittest.TestCase):

    def setUp(self):
        self.x = numpy.array(0.0, dtype=numpy.float32)
        self.y = numpy.array(1.0, dtype=numpy.float32)

    def check_different_eps(self, x, y):
        def f():
            if -1 < x < 1:
                return x.copy(),
            elif -2 < x < 2:
                return 2 * x,
            else:
                return 0,

        gx, = gradient_check.numerical_grad(f, (x,), (y,), eps=0.5)
        self.assertEqual(gx, 1.)
        gx, = gradient_check.numerical_grad(f, (x,), (y,), eps=1.5)
        self.assertEqual(gx, 2.)
        gx, = gradient_check.numerical_grad(f, (x,), (y,), eps=2.5)
        self.assertEqual(gx, 0.)

    def test_differenct_eps_cpu(self):
        self.check_different_eps(self.x, self.y)

    @attr.gpu
    def test_differenct_eps_gpu(self):
        self.check_different_eps(cuda.to_gpu(self.x), cuda.to_gpu(self.y))


default_eps = 1e-3


# `result`: True if `func` is non-differentiable on `x`
@testing.parameterize(*[
    {'func': 'zero', 'x': [-100.], 'result': False},
    {'func': 'zero', 'x': [100.], 'result': False},
    {'func': 'zero', 'x': [0.], 'result': False},
    {'func': 'zero', 'x': [default_eps / 10], 'result': False},
    {'func': 'zero', 'x': numpy.random.normal(size=(3, 2)), 'result': False},
    {'func': 'zero', 'x': numpy.random.normal(size=()), 'result': False},
    {'func': 'linear', 'x': [-100.], 'result': False},
    {'func': 'linear', 'x': [100.], 'result': False},
    {'func': 'linear', 'x': [0.], 'result': False},
    {'func': 'linear', 'x': numpy.random.normal(size=(3, 2)), 'result': False},
    {'func': 'linear', 'x': numpy.random.normal(size=()), 'result': False},
    # (Invalid input domain)
    {'func': 'linear', 'x': [numpy.inf], 'result': False,
     'ignore_warning': RuntimeWarning},
    {'func': 'quadratic', 'x': [-100.], 'result': False},
    {'func': 'quadratic', 'x': [100.], 'result': False},
    {'func': 'quadratic', 'x': [0.], 'result': False},
    {'func': 'cubic', 'x': [-100.], 'result': False},
    {'func': 'cubic', 'x': [100.], 'result': False},
    {'func': 'cubic', 'x': [0.], 'result': False},
    # Too large epsilon
    {'func': 'cubic', 'x': [0.], 'eps': 1e-1, 'result': True},
    {'func': 'abs', 'x': [0.], 'result': True},
    {'func': 'abs', 'x': [[3, 1], [0, 2]], 'result': True},
    {'func': 'abs', 'x': [default_eps * 0.8], 'result': True},
    {'func': 'abs', 'x': [-default_eps * 0.8], 'result': True},
    {'func': 'abs', 'x': [default_eps * 1.2], 'result': False},
    {'func': 'abs', 'x': [-default_eps * 1.2], 'result': False},
    {'func': 'abs', 'x': [100.], 'result': False},
    {'func': 'abs', 'x': [-100.], 'result': False},
    {'func': 'step', 'x': [0.], 'result': True},
    {'func': 'step', 'x': [default_eps * 0.8], 'result': True},
    {'func': 'step', 'x': [-default_eps * 0.8], 'result': True},
    {'func': 'step', 'x': [default_eps * 1.2], 'result': False},
    {'func': 'step', 'x': [-default_eps * 1.2], 'result': False},
    {'func': 'step', 'x': [100.], 'result': False},
    {'func': 'step', 'x': [-100.], 'result': False},
    {'func': 'clip', 'x': [0.], 'result': True},
    {'func': 'clip', 'x': [1.], 'result': True},
    {'func': 'clip', 'x': [0.5], 'result': False},
    {'func': 'floor', 'x': [0.], 'result': True},
    {'func': 'floor', 'x': [100 + default_eps * 0.8], 'result': True},
    {'func': 'floor', 'x': [100 - default_eps * 0.8], 'result': True},
    {'func': 'floor', 'x': [100 + default_eps * 1.2], 'result': False},
    {'func': 'floor', 'x': [100 - default_eps * 1.2], 'result': False},
    {'func': 'exp', 'x': [-100], 'result': False},
    {'func': 'exp', 'x': [0.], 'result': False},
    {'func': 'exp', 'x': [13.], 'result': False},
    {'func': 'log', 'x': [100.], 'result': False},
    # (Smaller epsilon is required because slope is steep)
    {'func': 'log', 'x': [1e-3], 'eps': 1e-6, 'result': False},
    {'func': 'log', 'x': [0.], 'result': True,
     'ignore_warning': RuntimeWarning},
    # (Invalid input domain)
    {'func': 'log', 'x': [-10.], 'result': False,
     'ignore_warning': RuntimeWarning},
    {'func': 'tan', 'x': [default_eps * 1.2], 'result': False},
    {'func': 'tan', 'x': [default_eps * 0.8], 'result': False},
    {'func': 'tan', 'x': [math.pi / 2], 'result': True},
    {'func': 'tan', 'x': [-math.pi / 2], 'result': True},
    {'func': 'tan', 'x': [3 * math.pi / 2], 'result': True},
    {'func': 'tan', 'x': [3 * math.pi / 2 + default_eps * 0.8],
     'result': True},
    {'func': 'tan', 'x': [3 * math.pi / 2 - default_eps * 0.8],
     'result': True},
    # (Smaller epsilon is required because slope is steep)
    {'func': 'tan', 'x': [3 * math.pi / 2 + 1e-3], 'eps': 1e-6,
     'result': False},
    # (Smaller epsilon is required because slope is steep)
    {'func': 'tan', 'x': [3 * math.pi / 2 - 1e-3], 'eps': 1e-6,
     'result': False},
    {'func': 'nan_segment', 'x': [0.], 'result': False},
    {'func': 'nan_segment', 'x': [-1.], 'result': True},
    {'func': 'nan_segment', 'x': [1.], 'result': True},
])
class NumericalGradientDetectNondifferentiableTest(unittest.TestCase):

    def setUp(self):
        self.eps = getattr(self, 'eps', default_eps)
        self.ignore_warning = getattr(self, 'ignore_warning', None)

    def _func_zero(self, x):
        xp = chainer.backend.get_array_module(x)
        return xp.zeros_like(x),

    def _func_linear(self, x):
        return 2 * x,

    def _func_quadratic(self, x):
        return x * x + 2.,

    def _func_cubic(self, x):
        return -3 * x ** 3 + 2 * x ** 2 + 1,

    def _func_abs(self, x):
        return abs(x),

    def _func_step(self, x):
        xp = chainer.backend.get_array_module(x)
        y = xp.zeros_like(x)
        y[x > 0] = 1
        return y,

    def _func_clip(self, x):
        y = x.clip(0, 1)
        return y,

    def _func_floor(self, x):
        xp = chainer.backend.get_array_module(x)
        return xp.floor(x),

    def _func_exp(self, x):
        xp = chainer.backend.get_array_module(x)
        return xp.exp(x),

    def _func_log(self, x):
        xp = chainer.backend.get_array_module(x)
        return xp.log(x),

    def _func_tan(self, x):
        xp = chainer.backend.get_array_module(x)
        return xp.tan(x),

    def _func_nan_segment(self, x):
        xp = chainer.backend.get_array_module(x)
        y = xp.ones_like(x)
        y[-1 < x < 1] = numpy.nan
        return y,

    def check_positive(self, xp, func_name, input, eps, nout):
        # Should be non-differentiable
        func = getattr(self, '_func_{}'.format(func_name))
        grad_outputs = [
            xp.random.uniform(-1, 1, input.shape).astype(input.dtype)
            for _ in range(nout)]

        def f():
            return func(input) * nout

        try:
            gradient_check.numerical_grad(
                f, (input,), grad_outputs, eps=eps,
                detect_nondifferentiable=True)
        except gradient_check.NondifferentiableError:
            pass
        else:
            raise AssertionError(
                'Function `{}` is expected to be non-differentiable, '
                'but determined to be differentiable.\n\n'
                'eps: {}\n'
                'input: {}\n'
                'xp: {}\n'
                ''.format(
                    func_name, eps, input, xp.__name__))

    def check_negative(self, xp, func_name, input, eps, nout):
        # Should be differentiable
        func = getattr(self, '_func_{}'.format(func_name))
        grad_outputs = [
            xp.random.uniform(-1, 1, input.shape).astype(input.dtype)
            for _ in range(nout)]

        def f():
            return func(input) * nout

        try:
            gradient_check.numerical_grad(
                f, (input,), grad_outputs, eps=eps,
                detect_nondifferentiable=True)
        except gradient_check.NondifferentiableError as e:
            raise AssertionError(
                'Function `{}` is expected to be differentiable, '
                'but determined to be non-differentiable.\n\n'
                'eps: {}\n'
                'input: {}\n'
                'xp: {}\n\n'
                '{}: {}'
                ''.format(
                    func_name, eps, input, xp.__name__,
                    e.__class__.__name__, e))

    def check(self, xp, nout):
        input = xp.asarray(self.x).astype(numpy.float32)
        with warnings.catch_warnings():
            if self.ignore_warning:
                warnings.simplefilter('ignore', self.ignore_warning)

            if self.result:
                self.check_positive(xp, self.func, input, self.eps, nout)
            else:
                self.check_negative(xp, self.func, input, self.eps, nout)

    def test_cpu(self):
        self.check(numpy, 1)

    @attr.gpu
    def test_gpu(self):
        self.check(cuda.cupy, 1)

    def test_2_outputs_cpu(self):
        self.check(numpy, 2)

    @attr.gpu
    def test_2_outputs_gpu(self):
        self.check(cuda.cupy, 2)


class AssertAllCloseTest(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.y = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

    def check_identical(self, x):
        testing.assert_allclose(x, x, atol=0, rtol=0)

    @condition.repeat(5)
    def test_identical_cpu(self):
        self.check_identical(self.x)

    @condition.repeat(5)
    @attr.gpu
    def test_identical_gpu(self):
        self.check_identical(cuda.to_gpu(self.x))

    def check_atol(self, x, y):
        x_cpu = cuda.to_cpu(x)
        y_cpu = cuda.to_cpu(y)
        max_abs_diff = numpy.max(numpy.abs(x_cpu - y_cpu))
        with self.assertRaises(AssertionError):
            testing.assert_allclose(x, y, atol=max_abs_diff - 1, rtol=0)
        testing.assert_allclose(x, y, atol=max_abs_diff + 1, rtol=0)

    @condition.repeat(5)
    def test_atol_cpu(self):
        self.check_atol(self.x, self.y)

    @condition.repeat(5)
    @attr.gpu
    def test_atol_gpu(self):
        self.check_atol(cuda.to_gpu(self.x), cuda.to_gpu(self.y))


class AssertAllCloseTest2(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.y = numpy.random.uniform(1, 2, (2, 3)).astype(numpy.float32)

    def check_rtol(self, x, y):
        x_cpu = cuda.to_cpu(x)
        y_cpu = cuda.to_cpu(y)
        max_ratio = numpy.max(numpy.abs(x_cpu - y_cpu) / y_cpu)
        with self.assertRaises(AssertionError):
            testing.assert_allclose(x, y, atol=0, rtol=max_ratio - 1)
        testing.assert_allclose(x, y, atol=0, rtol=max_ratio + 1)

    @condition.repeat(5)
    def test_rtol_cpu(self):
        self.check_rtol(self.x, self.y)

    @condition.repeat(5)
    @attr.gpu
    def test_rtol_gpu(self):
        self.check_rtol(cuda.to_gpu(self.x), cuda.to_gpu(self.y))


class Ident(chainer.Function):

    def forward(self, inputs):
        return inputs

    def backward(self, inputs, grads):
        return grads


# numpy.float16 is not tested because of the low precision.
@testing.parameterize(*testing.product({
    'dtype': [None, numpy.float32, numpy.float64],
}))
@backend.inject_backend_tests(None, [
    {},
    {'use_cuda': True},
    {'use_chainerx': True, 'chainerx_device': 'native:0'},
    {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
])
class TestCheckBackward(unittest.TestCase):

    def test_multiple_output(self, backend_config):
        x1 = backend_config.get_array(numpy.array([1], dtype='f'))
        x2 = backend_config.get_array(numpy.array([1], dtype='f'))
        g1 = backend_config.get_array(numpy.array([1], dtype='f'))
        g2 = backend_config.get_array(numpy.array([1], dtype='f'))

        def f(x, y):
            s, t = Ident()(x, y)
            u = Ident()(t)
            return s, u

        gradient_check.check_backward(
            f, (x1, x2), (g1, g2), dtype=self.dtype, atol=1e-4, rtol=1e-3)

    def test_no_grads_for_not_float(self, backend_config):
        if backend_config.use_chainerx:
            raise unittest.SkipTest(
                'gradient_check does not support no_grad option for ChainerX')
        x1 = backend_config.get_array(numpy.array([1], dtype='f'))
        # grad check for this is skipped
        x2 = backend_config.get_array(numpy.array([0, 1], dtype='i'))
        g1 = backend_config.get_array(numpy.array([1], dtype='f'))

        def f(x, y):
            # Integer data is not casted even when dtype is given
            self.assertEqual(y.dtype, 'i')
            s = Ident()(x)
            return s,

        gradient_check.check_backward(f, (x1, x2), g1, dtype=self.dtype)

    def test_no_grads_option(self, backend_config):
        if backend_config.use_chainerx:
            raise unittest.SkipTest(
                'gradient_check does not support no_grad option for ChainerX')
        x1 = backend_config.get_array(numpy.array([2], dtype='f'))
        # grad check for this is skipped
        x2 = backend_config.get_array(numpy.array([3], dtype='f'))
        g1 = backend_config.get_array(numpy.array([5], dtype='f'))

        def f(x, y):
            y_array = y.array
            if (backend_config.xp is chainerx
                    and isinstance(y_array, chainerx.ndarray)):
                y_array = y_array.as_grad_stopped()
            s = x + y_array
            return s,

        self.assertRaises(
            RuntimeError,  # backward computes x1.grad
            gradient_check.check_backward,
            f, (x1, x2), g1, no_grads=[True, True])

    def test_const_input(self, backend_config):
        x1 = backend_config.get_array(numpy.array([2], dtype='f'))
        # grad check for this is skipped
        x2 = backend_config.get_array(numpy.array([3], dtype='f'))
        g1 = backend_config.get_array(numpy.array([5], dtype='f'))

        def f(x, y):
            y_array = y.array
            if (backend_config.xp is chainerx
                    and isinstance(y_array, chainerx.ndarray)):
                y_array = y_array.as_grad_stopped()
            s = x + y_array
            return s,

        self.assertRaises(
            AssertionError,  # numerical backward to x2 is nonzero
            gradient_check.check_backward,
            f, (x1, x2), g1, no_grads=[False, False])

    def test_no_grads_option_with_dtype(self, backend_config):
        if backend_config.use_chainerx:
            raise unittest.SkipTest(
                'gradient_check does not support no_grad option for ChainerX')
        x1 = backend_config.get_array(numpy.array([1], dtype='f'))
        x2 = backend_config.get_array(numpy.array([1], dtype='f'))
        g1 = backend_config.get_array(numpy.array([1], dtype='f'))
        eps = 1e-3

        def f(x, y):
            if self.dtype is not None:
                # Check for correct dtypes if f is called to compute the
                # numerical gradient
                if x.data != x1:
                    self.assertEqual(x.dtype, self.dtype)
                    self.assertEqual(x.dtype, y.dtype)
            s = Ident()(x)
            return s,

        gradient_check.check_backward(f, (x1, x2), g1, eps=eps,
                                      no_grads=[False, True], dtype=self.dtype)


class IdentNoneIsZero(chainer.Function):
    """Identity function but following None-grad convention for RNNs"""

    def forward(self, inputs):
        return inputs

    def backward(self, inputs, grads):
        return tuple(
            numpy.zeros_like(x) if g is None else g
            for x, g in zip(inputs, grads)
        )


@testing.parameterize(*testing.product({
    'dtype': [None, numpy.float32, numpy.float64],
    'size': [3, 1]
}))
class TestCheckBackwardNoneConvention(unittest.TestCase):
    dtype = numpy.float64

    def test_multiple_output(self):
        size = self.size
        x1 = numpy.arange(size).astype('f')
        x2 = numpy.arange(size).astype('f')
        g1 = numpy.ones(size, dtype='f')
        g2 = numpy.ones(size, dtype='f')

        def f(x, y):
            s, t = IdentNoneIsZero()(x, y)
            return s, t

        gradient_check.check_backward(
            f, (x1, x2), (g1, g2), dtype=self.dtype, atol=1e-4, rtol=1e-3)
        gradient_check.check_backward(
            f, (x1, x2), (g1, None), dtype=self.dtype, atol=1e-4, rtol=1e-3)
        gradient_check.check_backward(
            f, (x1, x2), (None, g2), dtype=self.dtype, atol=1e-4, rtol=1e-3)


class TestCheckBackwardFailure(unittest.TestCase):

    def _broken_func_1(self):
        class Broken(chainer.Function):
            def forward(self, inputs):
                x, = inputs
                return (x * x),

            def backward(self, inputs, grad_outputs):
                x, = inputs
                gy, = grad_outputs
                return 3 * x * gy,

        return Broken()

    def _broken_func_2(self):
        class Broken(chainer.FunctionNode):
            def forward(self, inputs):
                x, = inputs
                self.retain_inputs((0,))
                return (x * x),

            def backward(self, indexes, grad_outputs):
                x, = self.get_retained_inputs()
                gy, = grad_outputs
                return 3 * x * gy,

        return Broken()

    def _broken_func_3(self):
        class Broken(chainer.FunctionNode):
            def forward(self, inputs):
                x, = inputs
                self.retain_inputs((0,))
                return (x * x),

            def backward(self, indexes, grad_outputs):
                x, = self.get_retained_inputs()
                gy, = grad_outputs
                gx1 = 2 * x * gy
                gx2 = 3 * x * gy
                return (gx1, gx2)

        return Broken()

    def test_fail_function(self):
        # Invalid backward (chainer.Function)
        x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

        def f(x):
            return self._broken_func_1()(x)

        with self.assertRaises(AssertionError):
            gradient_check.check_backward(f, x, gy)

    def test_fail_function_node(self):
        # Invalid backward (chainer.FunctionNode)
        x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

        def f(x):
            return self._broken_func_2().apply((x,))

        with self.assertRaises(AssertionError):
            gradient_check.check_backward(f, x, gy)

    def test_fail_invalid_number_of_gradients(self):
        # Invalid number of gradients
        x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)

        def f(x):
            return self._broken_func_3().apply((x,))

        with self.assertRaises(ValueError):
            gradient_check.check_backward(f, x, gy)

    def test_fail_invalid_number_of_gradients_0_size(self):
        # Invalid number of gradients (0-sized input)
        x = numpy.random.uniform(-1, 1, (2, 0)).astype(numpy.float32)
        gy = numpy.random.uniform(-1, 1, (2, 0)).astype(numpy.float32)

        def f(x):
            return self._broken_func_3().apply((x,))

        with self.assertRaises(ValueError):
            gradient_check.check_backward(f, x, gy)


class NewIdent(chainer.FunctionNode):

    def forward(self, inputs):
        return inputs

    def backward(self, indexes, grad_outputs):
        return NewIdent().apply(grad_outputs)


@backend.inject_backend_tests(None, [
    {},
    {'use_cuda': True},
    {'use_chainerx': True, 'chainerx_device': 'native:0'},
    {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
])
class TestCheckDoubleBackward(unittest.TestCase):

    def test_multiple_input_output(self, backend_config):
        x1, x2, gy1, gy2, ggx1, ggx2 = [
            backend_config.get_array(numpy.ones((2, 3), 'f'))
            for _ in range(6)]

        def f(x, y):
            w1 = x + y
            w2 = w1 + y
            return w1 * w1, w2 * w2

        gradient_check.check_double_backward(
            f, (x1, x2), (gy1, gy2),
            (ggx1, ggx2), dtype='d', atol=1e-3, rtol=1e-3)

    def test_double_backward_with_params(self, backend_config):
        if backend_config.use_chainerx:
            raise unittest.SkipTest(
                'ChainerX does not support params argument of '
                'gradient_check.check_double_backward().')
        x, gy, ggx, param_a, ggparam = [
            backend_config.get_array(numpy.ones((2, 3), 'f'))
            for _ in range(5)]

        param = chainer.Variable(param_a)

        def f(x):
            return x * param

        gradient_check.check_double_backward(
            f, x, gy, ggx, param, ggparam, atol=1e-3, rtol=1e-3)


testing.run_module(__name__, __file__)
