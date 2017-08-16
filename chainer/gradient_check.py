import warnings

import numpy
import six

from chainer import configuration
from chainer import cuda
from chainer.functions.math import identity
from chainer import testing
from chainer import variable


def _copy_arrays(xs):
    xp = cuda.get_array_module(*xs)
    return [xp.copy(x, order='C') for x in xs]


def numerical_grad(f, inputs, grad_outputs, eps=1e-3):
    """Computes numerical gradient by finite differences.

    This function is used to implement gradient check. For usage example, see
    unit tests of :mod:`chainer.functions`.

    Args:
        f (function): Python function with no arguments that runs forward
            computation and returns the result.
        inputs (tuple of arrays): Tuple of arrays that should be treated as
            inputs. Each element of them is slightly modified to realize
            numerical gradient by finite differences.
        grad_outputs (tuple of arrays): Tuple of arrays that are treated as
            output gradients.
        eps (float): Epsilon value of finite differences.

    Returns:
        tuple: Numerical gradient arrays corresponding to ``inputs``.

    """
    assert eps > 0
    inputs = tuple(inputs)
    grad_outputs = tuple(grad_outputs)
    gpu = any(isinstance(x, cuda.ndarray) for x in inputs + grad_outputs)
    cpu = any(isinstance(x, numpy.ndarray) for x in inputs + grad_outputs)

    if gpu and cpu:
        raise RuntimeError('Do not mix GPU and CPU arrays in `numerical_grad`')

    if gpu:
        xp = cuda.cupy
        numerical_grad_kernel = cuda.reduce(
            'T y1, T y2, U gy, T eps', 'V gxi',
            '(y1 - y2) * gy', 'a + b', 'gxi += a / (eps * 2)', '0',
            'numerical_grad_kernel'
        )
    else:
        xp = numpy
    grads = [xp.zeros_like(x) for x in inputs]

    with configuration.using_config('type_check', False):
        for x, gx in six.moves.zip(inputs, grads):
            for i in numpy.ndindex(x.shape):
                orig = x[i].copy()  # hold original value
                x[i] = orig + eps
                ys1 = _copy_arrays(f())
                x[i] = orig - eps
                ys2 = _copy_arrays(f())
                x[i] = orig
                for y1, y2, gy in six.moves.zip(ys1, ys2, grad_outputs):
                    if gy is not None:
                        if (gpu and isinstance(y1, cuda.ndarray) and
                                isinstance(y2, cuda.ndarray) and
                                isinstance(gy, cuda.ndarray)):
                            numerical_grad_kernel(y1, y2, gy, eps, gx[i])
                        else:
                            dot = ((y1 - y2) * gy).sum()
                            gx[i] += dot / (2 * eps)

    return grads


def assert_allclose(x, y, atol=1e-5, rtol=1e-4, verbose=True):
    """Asserts if some corresponding element of x and y differs too much.

    This function can handle both CPU and GPU arrays simultaneously.

    Args:
        x: Left-hand-side array.
        y: Right-hand-side array.
        atol (float): Absolute tolerance.
        rtol (float): Relative tolerance.
        verbose (bool): If ``True``, it outputs verbose messages on error.

    """
    warnings.warn(
        'chainer.gradient_check.assert_allclose is deprecated.'
        'Use chainer.testing.assert_allclose instead.',
        DeprecationWarning)
    testing.assert_allclose(x, y, atol, rtol, verbose)


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return x,


def check_backward(func, x_data, y_grad, params=(),
                   eps=1e-3, atol=1e-5, rtol=1e-4, no_grads=None, dtype=None):
    """Test backward procedure of a given function.

    This function automatically checks backward-process of a given function.
    For example, when you have a :class:`~chainer.Function` class ``MyFunc``,
    that gets two arguments and returns one value, you can make its test like
    this::

    >> def test_my_func(self):
    >>   func = MyFunc()
    >>   x1_data = xp.array(...)
    >>   x2_data = xp.array(...)
    >>   gy_data = xp.array(...)
    >>   check_backward(func, (x1_data, x2_data), gy_data)

    This method creates :class:`~chainer.Variable` objects with ``x_data``
    and calls ``func`` with the :class:`~chainer.Variable` s to get its result
    as :class:`~chainer.Variable`.
    Then, it sets ``y_grad`` array to ``grad`` attribute of the result and
    calls ``backward`` method to get gradients of the inputs.
    To check correctness of the gradients, the function calls
    :func:`numerical_grad` to calculate numerically the gradients and compares
    the types of gradients with :func:`chainer.testing.assert_allclose`.

    To reduce computational time, it uses a function
    :math:`g: \\mathbb{R} \\rightarrow \\mathbb{R}^n` defined as
    :math:`g(\\alpha) = f(\\alpha x)`, where :math:`\\alpha \\in \\mathbb{R}`
    and :math:`f` is a function which actually
    you want to test.
    Its gradient is

    .. math::
       g'(\\alpha) = f'(\\alpha x) \\cdot x.

    When :math:`\\alpha = 1`, :math:`g'(1) = f'(x) \\cdot x`.
    So :math:`g'(1)` is calculated with :func:`numerical_grad` and
    compared with dot product of the gradient :math:`f` and
    :math:`x`.

    If input objects (``x1_data`` or/and ``x2_data`` in this example) represent
    integer variables, their gradients are ignored.

    You can simplify a test when ``MyFunc`` gets only one argument::

    >>   check_backward(func, x1_data, gy_data)

    If ``MyFunc`` is a loss function which returns a zero-dimensional
    array, pass ``None`` to ``gy_data``. In this case, it sets ``1`` to
    ``grad`` attribute of the result::

    >>   check_backward(my_loss_func, (x1_data, x2_data), None)

    If ``MyFunc`` returns multiple outputs, pass all gradients for outputs
    as a tuple::

    >>   gy1_data = xp.array(...)
    >>   gy2_data = xp.array(...)
    >>   check_backward(func, x1_data, (gy1_data, gy2_data))

    You can also test a :class:`~chainer.Link`.
    To check gradients of parameters of the link, set a tuple of the parameters
    to ``params`` arguments::

    >>   check_backward(my_link, (x1_data, x2_data), gy_data,
    >>                  (my_link.W, my_link.b))

    Note that ``params`` are not ``ndarray`` s,
    but :class:`~chainer.Variables` s.

    Function objects are acceptable as ``func`` argument::

    >>   check_backward(lambda x1, x2: f(x1, x2),
    >>                  (x1_data, x2_data), gy_data)

    .. note::

       ``func`` is called many times to get numerical gradients for all inputs.
       This function doesn't work correctly when ``func`` behaves randomly as
       it gets different gradients.


    Args:
        func (callable): A function which gets :class:`~chainer.Variable` s
            and returns :class:`~chainer.Variable` s. ``func`` must returns
            a tuple of :class:`~chainer.Variable` s or one
            :class:`~chainer.Variable`. You can use :class:`~chainer.Function`
            object, :class:`~chainer.Link` object or a function satisfying the
            condition.
        x_data (ndarray or tuple of ndarrays): A set of ``ndarray`` s to be
            passed to ``func``. If ``x_data`` is one ``ndarray`` object, it is
            treated as ``(x_data,)``.
        y_grad (ndarray or tuple of ndarrays or None):
            A set of ``ndarray`` s representing gradients of return-values of
            ``func``. If ``y_grad`` is one ``ndarray`` object, it is
            treated as ``(y_grad,)``. If ``func`` is a loss-function,
            ``y_grad`` should be set to ``None``.
        params (~chainer.Variable or tuple of ~chainder.Variable):
            A set of :class:`~chainer.Variable` s whose gradients are checked.
            When ``func`` is a :class:`~chainer.Link` object,
            set its parameters as ``params``.
            If ``params`` is one :class:`~chainer.Variable` object,
            it is treated as ``(params,)``.
        eps (float): Epsilon value to be passed to :func:`numerical_grad`.
        atol (float): Absolute tolerance to be passed to
            :func:`chainer.testing.assert_allclose`.
        rtol (float): Relative tolerance to be passed to
            :func:`chainer.testing.assert_allclose`.
        no_grads (list of bool): Flag to skip variable for gradient assertion.
            It should be same length as ``x_data``.
        dtype (~numpy.dtype): ``x_data``, ``y_grad`` and ``params`` are casted
            to this dtype when calculating numerical gradients. Only float
            types and ``None`` are allowed.

    See:
       :func:`numerical_grad`
    """
    x_data = _as_tuple(x_data)
    if y_grad is not None:
        y_grad = _as_tuple(y_grad)
    params = _as_tuple(params)

    xs = [variable.Variable(x) for x in x_data]
    y = func(*xs)
    y = _as_tuple(y)

    # All creators of `y` need to be the same because we only call
    # `y[0].backward` to call `backward` method of the creator.
    # To do so we need to insert a dummy function `Ident` to the
    # computational graph.
    # Note that `func` may not be a `Function` object.
    y = identity.Identity()(*y)
    y = _as_tuple(y)

    if y_grad is not None:
        if len(y) != len(y_grad):
            raise ValueError(
                '`y_grad` must have the same length of output values')
        for iy, igy in six.moves.zip(y, y_grad):
            iy.grad = igy
    else:
        if len(y) != 1:
            raise ValueError(
                'When `y_grad` is `None`, the function must return a'
                'zero-dimentional array')
        y_grad = (1,)

    # We only need to call `backward` for one result `Variable`.
    # `Variable.backward` method calls `Function.backward` of its creator.
    y[0].backward()

    param_data = [p.data for p in params]
    if dtype is None:
        casted_xs = [variable.Variable(x) for x in x_data]
    else:
        if numpy.dtype(dtype).kind != 'f':
            raise ValueError('`dtype` is allowed only float type')
        casted_xs = [variable.Variable(x.astype(dtype, copy=False)
                                       if x.dtype.kind == 'f' else x)
                     for x in x_data]

    if no_grads is None:
        no_grads = [x.dtype.kind != 'f' for x in xs]
    else:
        if len(no_grads) != len(xs):
            raise ValueError(
                'Length of no_grads param and xs should be same.')
    casted_data = [x.data.copy() for x in casted_xs]
    for skip, x in six.moves.zip(no_grads, xs):
        if skip:
            assert x.grad is None
        else:
            if x.grad is None:
                raise RuntimeError(
                    'gradients of some arguments are not calculated')

    xp = cuda.get_array_module(*xs)
    one = xp.array(1., dtype)

    def g():
        # This functions is called twice in `numerical_grad`.
        # `one` is `1 + epsilon` or `1 - epsilon` in these calls.
        # See the document of `numerical_grad`.
        for skip, cx, data in six.moves.zip(no_grads, casted_xs, casted_data):
            if skip:
                continue
            # astype is require to store data with the given type
            data = (one * data).astype(data.dtype)
            if numpy.isscalar(data):
                data = xp.array(data)
            cx.data = data
        for param, data in six.moves.zip(params, param_data):
            if dtype is not None:
                param_dtype = dtype
            else:
                param_dtype = param.dtype
            # The inner astype is required to calculates __mul__ in
            # `param_type` when data is low accuracy float.
            # The outer one is require to store data with the given type.
            param.data = (one * data.astype(param_dtype)).astype(param_dtype)
        ys = func(*casted_xs)
        ys = _as_tuple(ys)
        ys_data = tuple(y.data for y in ys)
        for skip, cx, data in six.moves.zip(no_grads, casted_xs, casted_data):
            if skip:
                continue
            cx.data = data
        for param, data in six.moves.zip(params, param_data):
            param.data = data
        return ys_data

    gx, = numerical_grad(g, (one,), y_grad, eps=eps)
    gx_accum = 0
    for skip, x, cx in six.moves.zip(no_grads, xs, casted_xs):
        if skip:
            continue
        gxi = x.grad.ravel()
        cxi = cx.data.ravel()
        if dtype is not None:
            gxi = gxi.astype(dtype)
            cxi = cxi.astype(dtype)
        gx_accum += gxi.dot(cxi)

    for p in params:
        gpi = p.grad.ravel()
        pi = p.data.ravel()
        if dtype is not None:
            gpi = gpi.astype(dtype)
            pi = pi.astype(dtype)
        gx_accum += gpi.dot(pi)

    testing.assert_allclose(gx, gx_accum, atol=atol, rtol=rtol)
