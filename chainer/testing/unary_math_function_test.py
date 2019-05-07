import unittest
import warnings

import numpy

from chainer.backends import cuda
from chainer import function
from chainer import functions
from chainer import variable

try:
    from chainer.testing import attr
    _error = attr.get_error()
except ImportError as e:
    _error = e


def is_available():
    return _error is None


def check_available():
    if _error is not None:
        raise RuntimeError('''\
{} is not available.

Reason: {}: {}'''.format(__name__, type(_error).__name__, _error))


def _func_name(func):
    if isinstance(func, function.Function):
        return func.__class__.__name__.lower()
    else:
        return func.__name__


def _func_class(func):
    if isinstance(func, function.Function):
        return func.__class__
    else:
        name = func.__name__.capitalize()
        return getattr(functions, name, None)


def _make_data_default(shape, dtype):
    x = numpy.random.uniform(-1, 1, shape).astype(dtype, copy=False)
    gy = numpy.random.uniform(-1, 1, shape).astype(dtype, copy=False)
    ggx = numpy.random.uniform(-1, 1, shape).astype(dtype, copy=False)
    return x, gy, ggx


def _nonlinear(func):
    def aux(x):
        y = func(x)
        return y * y
    return aux


def unary_math_function_unittest(func, func_expected=None, label_expected=None,
                                 make_data=None, is_linear=None,
                                 forward_options=None,
                                 backward_options=None,
                                 double_backward_options=None):
    """Decorator for testing unary mathematical Chainer functions.

    This decorator makes test classes test unary mathematical Chainer
    functions. Tested are forward and backward, including double backward,
    computations on CPU and GPU across parameterized ``shape`` and ``dtype``.

    Args:
        func(function or ~chainer.Function): Chainer function to be tested by
            the decorated test class. Taking :class:`~chainer.Function` is for
            backward compatibility.
        func_expected: Function used to provide expected values for
            testing forward computation. If not given, a corresponsing numpy
            function for ``func`` is implicitly picked up by its name.
        label_expected(string): String used to test labels of Chainer
            functions. If not given, the name of ``func`` is implicitly used.
        make_data: Function to customize input and gradient data used
            in the tests. It takes ``shape`` and ``dtype`` as its arguments,
            and returns a tuple of input, gradient and double gradient data. By
            default, uniform destribution ranged ``[-1, 1]`` is used for all of
            them.
        is_linear: Tells the decorator that ``func`` is a linear function
            so that it wraps ``func`` as a non-linear function to perform
            double backward test. This argument is left for backward
            compatibility. Linear functions can be tested by default without
            specifying ``is_linear`` in Chainer v5 or later.
        forward_options(dict): Options to be specified as an argument of
            :func:`chainer.testing.assert_allclose` function.
            If not given, preset tolerance values are automatically selected.
        backward_options(dict): Options to be specified as an argument of
            :func:`chainer.gradient_check.check_backward` function.
            If not given, preset tolerance values are automatically selected
            depending on ``dtype``.
        double_backward_options(dict): Options to be specified as an argument
            of :func:`chainer.gradient_check.check_double_backward` function.
            If not given, preset tolerance values are automatically selected
            depending on ``dtype``.

    The decorated test class tests forward, backward and double backward
    computations on CPU and GPU across the following
    :func:`~chainer.testing.parameterize` ed parameters:

    - shape: rank of zero, and rank of more than zero
    - dtype: ``numpy.float16``, ``numpy.float32`` and ``numpy.float64``

    Additionally, it tests the label of the Chainer function.

    Chainer functions tested by the test class decorated with the decorator
    should have the following properties:

    - Unary, taking one parameter and returning one value
    - ``dtype`` of input and output are the same
    - Elementwise operation for the supplied ndarray

    .. admonition:: Example

       The following code defines a test class that tests
       :func:`~chainer.functions.sin` Chainer function, which takes a parameter
       with ``dtype`` of float and returns a value with the same ``dtype``.

       .. doctest::

          >>> import unittest
          >>> from chainer import testing
          >>> from chainer import functions as F
          >>>
          >>> @testing.unary_math_function_unittest(F.sin)
          ... class TestSin(unittest.TestCase):
          ...     pass

       Because the test methods are implicitly injected to ``TestSin`` class by
       the decorator, it is enough to place ``pass`` in the class definition.

       To customize test data, ``make_data`` optional parameter can be used.
       The following is an example of testing ``sqrt`` Chainer function, which
       is tested in positive value domain here instead of the default input.

       .. doctest::

          >>> import numpy
          >>>
          >>> def make_data(shape, dtype):
          ...     x = numpy.random.uniform(0.1, 1, shape).astype(dtype)
          ...     gy = numpy.random.uniform(-1, 1, shape).astype(dtype)
          ...     ggx = numpy.random.uniform(-1, 1, shape).astype(dtype)
          ...     return x, gy, ggx
          ...
          >>> @testing.unary_math_function_unittest(F.sqrt,
          ...                                       make_data=make_data)
          ... class TestSqrt(unittest.TestCase):
          ...     pass

       ``make_data`` function which returns input, gradient and double gradient
       data generated in proper value domains with given ``shape`` and
       ``dtype`` parameters is defined, then passed to the decorator's
       ``make_data`` parameter.

    """
    check_available()

    # TODO(takagi) In the future, the Chainer functions that could be tested
    #     with the decorator would be extended as:
    #
    #     - Multiple input parameters
    #     - Multiple output values
    #     - Other types than float: integer
    #     - Other operators other than analytic math: basic math

    # Import here to avoid mutual import.
    from chainer import gradient_check
    from chainer import testing

    is_new_style = not isinstance(func, function.Function)

    func_name = _func_name(func)
    func_class = _func_class(func)

    if func_expected is None:
        try:
            func_expected = getattr(numpy, func_name)
        except AttributeError:
            raise ValueError('NumPy has no functions corresponding '
                             'to Chainer function \'{}\'.'.format(func_name))

    if label_expected is None:
        label_expected = func_name
    elif func_class is None:
        raise ValueError('Expected label is given even though Chainer '
                         'function does not have its label.')

    if make_data is None:
        if is_new_style:
            make_data = _make_data_default
        else:
            def aux(shape, dtype):
                return _make_data_default(shape, dtype)[0:2]
            make_data = aux

    if is_linear is not None:
        warnings.warn('is_linear option is deprecated', DeprecationWarning)

    def f(klass):
        assert issubclass(klass, unittest.TestCase)

        def setUp(self):
            if is_new_style:
                self.x, self.gy, self.ggx = make_data(self.shape, self.dtype)
            else:
                self.x, self.gy = make_data(self.shape, self.dtype)

            if self.dtype == numpy.float16:
                self.forward_options = {
                    'atol': numpy.finfo('float16').eps,  # = 0.000977
                    'rtol': numpy.finfo('float16').eps,  # = 0.000977
                }
                self.backward_options = {
                    'eps': 2 ** -4, 'atol': 2 ** -4, 'rtol': 2 ** -4,
                    'dtype': numpy.float64}
                self.double_backward_options = {
                    'eps': 2 ** -4, 'atol': 2 ** -4, 'rtol': 2 ** -4,
                    'dtype': numpy.float64}
            else:
                self.forward_options = {'atol': 1e-4, 'rtol': 1e-4}
                self.backward_options = {
                    'dtype': numpy.float64, 'atol': 1e-4, 'rtol': 1e-4}
                self.double_backward_options = {
                    'dtype': numpy.float64, 'atol': 1e-4, 'rtol': 1e-4}
            if forward_options is not None:
                self.forward_options.update(forward_options)
            if backward_options is not None:
                self.backward_options.update(backward_options)
            if double_backward_options is not None:
                self.double_backward_options.update(double_backward_options)
        setattr(klass, 'setUp', setUp)

        def check_forward(self, x_data):
            x = variable.Variable(x_data)
            y = func(x)
            self.assertEqual(y.data.dtype, x_data.dtype)
            y_expected = func_expected(cuda.to_cpu(x_data), dtype=x_data.dtype)
            testing.assert_allclose(y_expected, y.data, **self.forward_options)
        setattr(klass, 'check_forward', check_forward)

        def test_forward_cpu(self):
            self.check_forward(self.x)
        setattr(klass, 'test_forward_cpu', test_forward_cpu)

        @attr.gpu
        def test_forward_gpu(self):
            self.check_forward(cuda.to_gpu(self.x))
        setattr(klass, 'test_forward_gpu', test_forward_gpu)

        def check_backward(self, x_data, y_grad):
            gradient_check.check_backward(
                func, x_data, y_grad, **self.backward_options)
        setattr(klass, 'check_backward', check_backward)

        def test_backward_cpu(self):
            self.check_backward(self.x, self.gy)
        setattr(klass, 'test_backward_cpu', test_backward_cpu)

        @attr.gpu
        def test_backward_gpu(self):
            self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))
        setattr(klass, 'test_backward_gpu', test_backward_gpu)

        if is_new_style:
            def check_double_backward(self, x_data, y_grad, x_grad_grad):
                func1 = _nonlinear(func) if is_linear else func
                gradient_check.check_double_backward(
                    func1, x_data, y_grad,
                    x_grad_grad, **self.double_backward_options)
            setattr(klass, 'check_double_backward', check_double_backward)

            def test_double_backward_cpu(self):
                self.check_double_backward(self.x, self.gy, self.ggx)
            setattr(klass, 'test_double_backward_cpu',
                    test_double_backward_cpu)

            @attr.gpu
            def test_double_backward_gpu(self):
                self.check_double_backward(
                    cuda.to_gpu(self.x), cuda.to_gpu(self.gy),
                    cuda.to_gpu(self.ggx))
            setattr(klass, 'test_double_backward_gpu',
                    test_double_backward_gpu)

        if func_class is not None:
            def test_label(self):
                self.assertEqual(func_class().label, label_expected)
            setattr(klass, 'test_label', test_label)

        # Return parameterized class.
        return testing.parameterize(*testing.product({
            'shape': [(3, 2), ()],
            'dtype': [numpy.float16, numpy.float32, numpy.float64]
        }))(klass)
    return f
