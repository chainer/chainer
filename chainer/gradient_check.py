from __future__ import absolute_import
import math
import warnings

import numpy
import six

import chainer
from chainer import backend
from chainer.backends import _cpu
from chainer.backends import cuda
from chainer import configuration
from chainer import FunctionNode
from chainer import testing
from chainer import utils
from chainer import variable
import chainerx


class NondifferentiableError(Exception):
    pass


def _copy_arrays(xs):
    xp = backend.get_array_module(*xs)
    if xp is chainerx:
        return [
            None if x is None
            else xp.array(x, dtype=numpy.float64, copy=True, device=x.device)
            for x in xs]
    else:
        return [xp.array(x, dtype=numpy.float64, copy=True) for x in xs]


def _ones_like(arr):
    device = cuda.get_device_from_array(arr)
    xp = backend.get_array_module(arr)
    with device:
        return xp.ones_like(arr)


def _make_outputs_props_in_error_message(outputs, grad_outputs):
    return (
        'Output shapes and dtypes         : {}\n'
        'Output gradient shapes and dtypes: {}'.format(
            utils._format_array_props(outputs),
            utils._format_array_props(grad_outputs)))


def _check_outputs_and_grad_outputs(outputs, grad_outputs):
    if len(outputs) != len(grad_outputs):
        raise ValueError(
            'Output gradients must contain equally as many elements as '
            'the number of output elements.\n'
            '{}'.format(
                _make_outputs_props_in_error_message(outputs, grad_outputs)))

    shapes_match = True
    dtypes_match = True
    for y, gy in zip(outputs, grad_outputs):
        if gy is None:
            continue
        if y is None and (gy == 0).all():
            continue
        if y.shape != gy.shape:
            shapes_match = False
        if y.dtype != gy.dtype:
            dtypes_match = False

    if not (shapes_match and dtypes_match):
        raise ValueError(
            'Shapes and/or dtypes of outputs and output gradients do not '
            'match.\n'
            '{}'.format(
                _make_outputs_props_in_error_message(outputs, grad_outputs)))


def numerical_grad(
        f, inputs, grad_outputs, eps=1e-3,
        detect_nondifferentiable=False, diff_atol=0, diff_rtol=1e-2,
        center_outputs=None):
    """Computes numerical gradient by finite differences.

    This function is used to implement gradient check. For usage example, see
    unit tests of :mod:`chainer.functions`.

    By default, ``numerical_grad`` computes the gradient to the first order of
    ``eps``.

    Args:
        f (callable): Python function with no arguments that runs forward
            computation and returns the result.
        inputs (tuple of arrays): Tuple of arrays that should be treated as
            inputs. Each element of them is slightly modified to realize
            numerical gradient by finite differences.
        grad_outputs (tuple of arrays or scalars): Tuple of arrays or scalars
            that are treated as output gradients.
        eps (float): Epsilon value of finite differences.
        detect_nondifferentiable (bool):
            ``False`` by default.
            If ``True``, ``numerical_grad`` checks whether ``f`` is
            differentiable at ``inputs``.
            It requires evaluation of ``f`` at 5 points instead of 2.
            As a side effect, the accuracy of numerical gradient will be
            increased to the third order of ``eps``.
            If it turns out that ``f`` is non-differentiable at ``input``,
            ``numerical_grad`` raises
            :class:`~chainer.gradient_check.NondifferentiableError`.
        diff_atol (float):
            Absolute tolerance of fitting error of non-differentiable point
            detection.
        diff_rtol (float):
            Tolerance of fitting error of non-differentiable point detection
            relative to the output values of ``f``.
        center_outputs (tuple of arrays or None):
            Only used if ``detect_nondifferentiable`` is ``True``.
            If specified, these arrays are used as the outputs of ``f`` at
            ``inputs``.
            Otherwise, it is calculated.
            It can be used to reduce the computation if these arrays are
            already calculated before calling ``numerical_grad``.

    Returns:
        tuple: Numerical gradient arrays corresponding to ``inputs``.

    """
    # TODO(niboshi): Deprecate `center_outputs` argument.
    # If dtype of this argument is not float64, often the resolution is
    # insufficient for numerical gradient calculation. We might use it only
    # when its dtype is float64, but it would be better to simply remove it.
    center_outputs = None

    assert eps > 0
    assert isinstance(inputs, (tuple, list))
    for x in inputs:
        if x.dtype.kind != 'f':
            raise RuntimeError(
                'The dtype of input arrays must be kind of float')

    inputs = tuple(inputs)
    # Cast grad_outputs to float64
    grad_outputs = tuple([
        None if g is None
        else numpy.float64(g) if numpy.isscalar(g)
        else g.astype(numpy.float64)
        for g in grad_outputs])

    if not chainer.is_arrays_compatible(
            [a for a in inputs + grad_outputs if not numpy.isscalar(a)]):
        raise RuntimeError('Do not mix GPU and CPU arrays in `numerical_grad`')

    device = backend.get_device_from_array(*(inputs + grad_outputs))
    xp = device.xp

    if xp is cuda.cupy:
        numerical_grad_kernel_1 = cuda.reduce(
            'T y1, T y2, U gy, T eps', 'V gxi',
            '(y1 - y2) * gy', 'a + b', 'gxi += a / (eps * 2)', '0',
            'numerical_grad_kernel_1'
        )
        numerical_grad_kernel_3 = cuda.reduce(
            'T y1, T y2, T y3, T y4, U gy, T eps', 'V gxi',
            '(-y1 + 8 * y2 - 8 * y3 + y4) * gy',
            'a + b', 'gxi += a / (eps * 6)', '0',
            'numerical_grad_kernel_3'
        )

    if xp is chainerx:
        grads = [
            xp.zeros(x.shape, numpy.float64, device=x.device) for x in inputs]
    else:
        grads = [xp.zeros(x.shape, numpy.float64) for x in inputs]

    if detect_nondifferentiable:
        if center_outputs is None:
            ys0 = _copy_arrays(f())
        else:
            ys0 = center_outputs
        nout = len(ys0)
        shapes = [y0.shape for y0 in ys0]
        sizes = numpy.array([y0.size for y0 in ys0])
        cumsizes = numpy.cumsum(sizes)

    # Evaluate func at a single input
    def eval_func(x, x_ind, delta, orig):
        x[x_ind] = orig + delta
        ys = _copy_arrays(f())
        assert len(ys) == len(grad_outputs)
        assert all([
            gy is None
            for y, gy in zip(ys, grad_outputs)
            if y is None])
        assert all([
            gy is None or numpy.isscalar(gy) or y.shape == gy.shape
            for y, gy in zip(ys, grad_outputs)])
        x[x_ind] = orig
        return ys

    # An iteration on a single input displacement
    def iterate_single_input(i_in, x, orig_x, x_ind):
        orig = orig_x[x_ind]
        # `yss` holds a list of output arrays for each of 2 or 5 sampling
        # points.
        if detect_nondifferentiable:
            yss = [
                eval_func(x, x_ind, -eps * 1., orig),
                eval_func(x, x_ind, -eps * .5, orig),
                ys0,
                eval_func(x, x_ind, +eps * .5, orig),
                eval_func(x, x_ind, +eps * 1., orig),
            ]
        else:
            yss = [
                eval_func(x, x_ind, -eps * 1, orig),
                eval_func(x, x_ind, +eps * 1, orig),
            ]

        assert all([
            y is None
            or (y.shape == yss[0][i].shape and y.dtype == yss[0][i].dtype)
            for ys in yss
            for i, y in enumerate(ys)])

        # If all the outputs are 0-size, skip non-differentiable check.
        if all([y is None or y.size == 0 for y in yss[0]]):
            detect_nondifferentiable_ = False
        else:
            detect_nondifferentiable_ = detect_nondifferentiable

        if detect_nondifferentiable_:
            # Detect non-differentiable point by quadratic fitting

            # Check for non-finite output.
            # If any single element in the output arrays has different
            # finiteness among sampled points, that means this is a
            # non-differentiable point.
            # If the function consistently generates non-finite values
            # around the point, we do not treat the point as
            # non-differentiable.
            # (Example: x<0 region for the logarithm function)
            any_nonfinite = False
            for i_out in range(nout):
                isfinites = [xp.isfinite(ys[i_out]) for ys in yss]
                if any((isfinites[0] != isfinites[i]).any()
                       for i in range(1, len(yss))):
                    s = six.StringIO()
                    s.write(
                        'Tried to compute the numeric gradient on a '
                        'non-differentiable point.\n\n')
                    s.write('i_in: {}\n'.format(i_in))
                    s.write('i_out: {}\n'.format(i_out))
                    s.write('x: {}\n'.format(inputs[i_in]))
                    s.write('index on x: {}\n'.format(x_ind))
                    s.write('eps: {}\n'.format(eps))
                    s.write('y[x-eps  ]: {}\n'.format(yss[0][i_out]))
                    s.write('y[x-eps/2]: {}\n'.format(yss[1][i_out]))
                    s.write('y[x      ]: {}\n'.format(yss[2][i_out]))
                    s.write('y[x+eps/2]: {}\n'.format(yss[3][i_out]))
                    s.write('y[x+eps  ]: {}\n'.format(yss[4][i_out]))
                    raise NondifferentiableError(s.getvalue())

                any_nonfinite |= not all((_).all() for _ in isfinites)

            if not any_nonfinite:
                # Stack flattened outputs to make (5, *)-shaped 2D array
                ystack = xp.vstack(
                    [xp.hstack([y.ravel() for y in ys]) for ys in yss])
                assert ystack.ndim == 2 and ystack.shape[0] == len(yss)
                # Fit to quadratic
                if xp is not numpy:
                    ystack = _cpu._to_cpu(ystack)
                polyfit = numpy.polynomial.polynomial.polyfit
                _, (residuals, _, _, _) = polyfit(
                    range(len(yss)), ystack, deg=2, full=True)
                if xp is not numpy:
                    residuals = device.send(residuals)
                residuals = xp.sqrt(residuals / len(yss))

                # Check for error for each output array
                for i_out in range(nout):
                    size = sizes[i_out]
                    cumsize = cumsizes[i_out]
                    shape = shapes[i_out]
                    # TODO(niboshi): The following two lines could be
                    # rewritten using xp.stack, which is supported in
                    # NumPy>=1.10
                    ymax = xp.concatenate(
                        [ys[i_out][None] for ys in yss]).max(axis=0)
                    ymin = xp.concatenate(
                        [ys[i_out][None] for ys in yss]).min(axis=0)
                    # Restore the shape of flattened residual
                    res = residuals[cumsize - size:cumsize]
                    res = res.reshape(shape)
                    det = utils.force_array(
                        diff_atol + diff_rtol * (ymax - ymin) < res)
                    # Constant output = not nondifferentiable
                    det[ymax == ymin] = False
                    if det.any():
                        s = six.StringIO()
                        s.write(
                            'Tried to compute the numeric gradient on a '
                            'non-differentiable point.\n\n')
                        s.write('i_in: {}\n'.format(i_in))
                        s.write('i_out: {}\n'.format(i_out))
                        s.write('x: {}\n'.format(inputs[i_in]))
                        s.write('index on x: {}\n'.format(x_ind))
                        s.write('eps: {}\n'.format(eps))
                        s.write('diff_rtol: {}\n'.format(diff_rtol))
                        s.write('diff_atol: {}\n'.format(diff_atol))
                        s.write('ymax: {}\n'.format(ymax))
                        s.write('ymin: {}\n'.format(ymin))
                        s.write(
                            'diff_atol + diff_rtol * (ymax-ymin): {}\n'.format(
                                diff_atol + diff_rtol * (ymax - ymin)))
                        s.write('fitting errors: {}\n'.format(res))
                        s.write('y[x-eps  ]: {}\n'.format(yss[0][i_out]))
                        s.write('y[x-eps/2]: {}\n'.format(yss[1][i_out]))
                        s.write('y[x      ]: {}\n'.format(yss[2][i_out]))
                        s.write('y[x+eps/2]: {}\n'.format(yss[3][i_out]))
                        s.write('y[x+eps  ]: {}\n'.format(yss[4][i_out]))
                        raise NondifferentiableError(s.getvalue())

        # Calculate numerical gradient
        for i_out, gy in enumerate(grad_outputs):
            if gy is None:
                continue
            if not numpy.isscalar(gy):
                gy = gy.astype(numpy.float64, copy=False)
            gpu_ = (xp is cuda.cupy and
                    all(isinstance(ys[i_out], cuda.ndarray)
                        for ys in yss))
            # If any output sample is None, all others must be.
            assert all([
                (yss[0][i_out] is None) == (yss[j][i_out] is None)
                for j in range(len(yss))])
            # If outputs samples are None, the part of numeric gradient for
            # this output is considered as zero: skip the accumulation.
            if yss[0][i_out] is None:
                continue

            if len(yss) == 2:  # 1st order
                y0 = yss[0][i_out]
                y1 = yss[1][i_out]
                if gpu_:
                    numerical_grad_kernel_1(
                        y1, y0, xp.asarray(gy), eps, gx[x_ind])
                else:
                    dot = ((y1 - y0) * gy).sum()
                    gx[x_ind] = gx[x_ind] + dot / (2 * eps)
            elif len(yss) == 5:  # 3rd order
                y0 = yss[0][i_out]
                y1 = yss[1][i_out]
                y2 = yss[3][i_out]
                y3 = yss[4][i_out]
                if gpu_:
                    numerical_grad_kernel_3(
                        y3, y2, y1, y0, gy, eps, gx[x_ind])
                else:
                    num = -y3 + 8 * y2 - 8 * y1 + y0
                    dot = (num * gy).sum()
                    gx[x_ind] = gx[x_ind] + dot / (6 * eps)
            else:
                assert False

    # Calculate numeric gradient
    with configuration.using_config('type_check', False):
        for i_in, (x, gx) in enumerate(six.moves.zip(inputs, grads)):
            orig_x = x.copy()  # hold original value
            for x_ind in numpy.ndindex(x.shape):
                iterate_single_input(i_in, x, orig_x, x_ind)

    return [g.astype(x.dtype, copy=False)
            for g, x in six.moves.zip(grads, inputs)]


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


def _as_tuple(xs):
    if isinstance(xs, tuple):
        return xs
    elif isinstance(xs, list):
        return tuple(xs)
    else:
        return xs,


class _CheckBackward(object):

    def __init__(
            self, func, xs, gys, params, eps, atol, rtol, no_gxs,
            dtype, detect_nondifferentiable, is_immutable_params):
        # If `is_immutable_params` is `False`, `params` are expected to be of
        # type `chainer.Parameter` and are updated in-place.
        # To run `_CheckBackward` with ChainerX ndarrays however which cannot
        # be updated in-place when wrapped in `chainer.Parameter`s, this flag
        # should be `True` and parameters should be given as ndarrays.
        # `func` in the former case must take inputs as arguments only. In the
        # latter, it must take the parameters in addition.

        if dtype is not None and numpy.dtype(dtype).kind != 'f':
            raise ValueError('`dtype` is allowed only float type')
        if is_immutable_params:
            if not all(
                    isinstance(p, chainer.get_array_types()) for p in params):
                raise ValueError(
                    'All parameters in `params` must be ndarrays if '
                    '`is_immutable_params` is `True`. Actual: {}.'.format(
                        ', '.join(str(type(p)) for p in params)))

        xs = _as_tuple(xs)
        if gys is not None:
            gys = _as_tuple(gys)
        params = _as_tuple(params)

        if no_gxs is None:
            no_gxs = [x.dtype.kind != 'f' for x in xs]
        else:
            if len(no_gxs) != len(xs):
                raise ValueError(
                    'Length of no_grads param and xs should be same.\n'
                    'Actual: {0} != {1}'.format(len(no_gxs), len(xs)))

        device = backend.get_device_from_array(*xs)

        if device.xp is chainerx:
            if params and not is_immutable_params:
                raise NotImplementedError(
                    'gradient_check does not support params argument for '
                    'ChainerX arrays')

        self.device = device

        self.func = func
        self.xs = xs
        self.gys = gys
        self.params = params
        self.no_gxs = no_gxs
        self.atol = atol
        self.rtol = rtol
        self.is_immutable_params = is_immutable_params
        # options for numeric gradients
        self.eps = eps
        self.dtype = dtype
        self.detect_nondifferentiable = detect_nondifferentiable

    def run(self):
        with chainer.using_device(self.device):
            self._run()

    def _run(self):
        # Run a forward pass for backward gradients.
        # Uninitialized parameters may be initialized.
        # If self.gys is None, it is also updated with 1s.
        # This must be done before sampling a direction vector, because
        # otherwise the shapes of uninitialized parameters wouldn't be
        # determined.
        xs_backward, ys, params_backward = (
            self._forward_for_backward_gradients())
        # Keep output arrays to save computation in numerical gradients
        ys0 = tuple([None if y is None else y.array for y in ys])

        # If gys is not given, generate the all-1 gradients.
        if self.gys is None:
            if not (len(ys) == 1 and ys[0].shape == ()):
                raise ValueError(
                    'y_grad argument cannot be omitted if the target function '
                    'is not a loss function, which has a single output with '
                    'shape ().\n'
                    'Actual output shapes: {}'.format(
                        ', '.join([str(y.shape) for y in ys])))

            self.gys = tuple([_ones_like(y.array) for y in ys])
        else:
            _check_outputs_and_grad_outputs(ys, self.gys)

        # Strike out gys corresponding to None y
        self.gys = tuple([
            None if y is None else gy for gy, y in zip(self.gys, ys0)])

        # Sample a direction vector.
        directions = self._sample_directions()

        # Compute backward gradients by running a backward pass.
        gx_backward = self._directional_backward_gradients(
            xs_backward, ys, params_backward, directions)

        # Compute numeric gradients
        gx_numeric = self._directional_numeric_gradients(directions, ys0)

        # Compare the resulted gradients
        self._compare_gradients(gx_numeric, gx_backward, directions)

    def _compare_gradients(self, gx_numeric, gx_backward, directions):
        atol = self.atol
        rtol = self.rtol
        # Compare the gradients
        try:
            testing.assert_allclose(
                gx_numeric, gx_backward, atol=atol, rtol=rtol)
        except AssertionError as e:
            eps = self.eps
            xs = self.xs
            gys = self.gys
            f = six.StringIO()
            f.write('check_backward failed (eps={} atol={} rtol={})\n'.format(
                eps, atol, rtol))
            for i, x in enumerate(xs):
                f.write('inputs[{}]:\n'.format(i))
                f.write('{}\n'.format(x))
            for i, gy in enumerate(gys):
                f.write('grad_outputs[{}]:\n'.format(i))
                f.write('{}\n'.format(gy))
            for i, d in enumerate(directions):
                f.write('directions[{}]:\n'.format(i))
                f.write('{}\n'.format(d))
            f.write('gradients (numeric):  {}\n'.format(gx_numeric))
            f.write('gradients (backward): {}\n'.format(gx_backward))
            f.write('\n')
            f.write('x: numeric gradient, y: backward gradient')
            f.write(str(e))
            raise AssertionError(f.getvalue())

    def _sample_directions(self):
        # Samples a direction vector (list of arrays with the same shapes as
        # input arrays and parameters)
        device = self.device
        xs = self.xs
        params = self.params
        no_gxs = self.no_gxs

        xp = device.xp
        direction_xs_shapes = [
            x.shape
            for x, no_gx in six.moves.zip(xs, no_gxs)
            if not no_gx]
        direction_param_shapes = [p.shape for p in params]
        direction_shapes = direction_xs_shapes + direction_param_shapes
        directions = [
            xp.random.normal(size=shape) for shape in direction_shapes]

        # The direction vector is normalized in order to keep the scale of
        # differentiation error invariant with respect to the number of input
        # dimensions. Ideally, the scale of the curvature with respect to each
        # input dimension should be taken into account, but we ignore the
        # differences and assume that the curvature is uniform with respect to
        # all the input dimensions.
        norm = math.sqrt(sum([xp.square(d).sum() for d in directions]))
        if norm != 0:
            # norm could be zero if input arrays are 0-sized.
            scale = 1. / norm
            directions = [d * scale for d in directions]

        return directions

    def _clear_grads(self, xs):
        for x in xs:
            x.grad_var = None

    def _forward_for_backward_gradients(self):
        func = self.func
        xs = self.xs
        params = self.params

        xs = [
            variable.Variable(x, requires_grad=x.dtype.kind == 'f')
            for x in xs]

        if self.is_immutable_params:
            params = tuple([chainer.Parameter(p) for p in params])
            ys = func(xs, params)
        else:
            ys = func(*xs)

        ys = _as_tuple(ys)

        # Clear gradients which may exist if func calls backward inside of
        # itself.
        self._clear_grads(xs)
        self._clear_grads(params)

        return xs, ys, params

    def _directional_backward_gradients(self, xs, ys, params, directions):
        no_gxs = self.no_gxs

        # We need to start backprop from a single variable,
        # so a dummy function `_GradientSetter` is inserted at the end of the
        # computational graph.
        y_backward = _apply_grad_setter_func(
            ys,
            [None if gy is None
             # Copy is needed to avoid being updated during backprop, which
             # would affect the numerical gradient.
             # TODO(niboshi): Preserve strides, for testing purpose.
             else chainer.Variable(gy.copy(), requires_grad=False)
             for gy in self.gys])

        # Backward
        y_backward.backward()

        for no_gx, x in six.moves.zip(no_gxs, xs):
            if no_gx and x.grad is not None:
                raise RuntimeError(
                    'gradient of int variable must be None')

        grads = (
            [x.grad for x, no_gx in six.moves.zip(xs, no_gxs)
             if not no_gx]
            + [p.grad for p in params])

        gx_accum = 0
        assert len(grads) == len(directions)
        for g, direction in six.moves.zip(grads, directions):
            if g is not None:
                gx_accum += (g.astype(numpy.float64) * direction).sum()

        return gx_accum

    def _directional_numeric_gradients(self, directions, y0_data):
        device = self.device
        func = self.func
        xs = self.xs
        gys = self.gys
        params = self.params
        eps = self.eps
        no_gxs = self.no_gxs
        dtype = self.dtype
        detect_nondifferentiable = self.detect_nondifferentiable
        params_data = [
            p if self.is_immutable_params else p.array for p in params]

        xp = device.xp

        x_vars = [variable.Variable(x, requires_grad=False) for x in xs]

        xs_filtered = [
            x.array for x, no_gx in six.moves.zip(x_vars, no_gxs) if not no_gx]

        if dtype is None:
            casted_data = [x for x in xs_filtered + params_data]
        else:
            if numpy.dtype(dtype).kind != 'f':
                raise ValueError('`dtype` is allowed only float type')

            # Even skipped variable must have the same dtype.
            for x, no_gx in six.moves.zip(x_vars, no_gxs):
                if no_gx and x.array.dtype.kind == 'f':
                    x.array = x.array.astype(dtype, copy=False)

            casted_data = [
                x.astype(dtype, copy=False)
                for x in xs_filtered + params_data]

        delta = xp.array(0., numpy.float64)

        def g():
            # This functions is called twice in `numerical_grad`.
            # `delta` is `epsilon` or `-epsilon` in these calls.
            # See the document of `numerical_grad`.

            def perturb(data, direction):
                data = (data.astype(numpy.float64)
                        + delta * direction).astype(data.dtype)
                if numpy.isscalar(data):
                    data = xp.array(data)
                return data

            # Input arrays
            g_x_vars = []
            j = 0
            for x_var, no_gx in six.moves.zip(x_vars, no_gxs):
                if no_gx:
                    g_x_vars.append(x_var)
                else:
                    data = perturb(casted_data[j], directions[j])
                    g_x_vars.append(variable.Variable(data))
                    j += 1

            # Parameters
            for i in range(len(params)):
                data = perturb(casted_data[j + i], directions[j + i])

                if self.is_immutable_params:
                    # Update the parameter array since it is converted into
                    # a Parameter just before calling the func.
                    params_data[i] = data
                else:
                    # Update the given Parameter in-place since the object is
                    # held by the caller.
                    params[i].array = data

            # Clear gradients to support func that calls backward inside of
            # itself.
            self._clear_grads(g_x_vars)
            if not self.is_immutable_params:
                self._clear_grads(params)

            if self.is_immutable_params:
                ps = tuple([chainer.Parameter(p) for p in params_data])
                ys = func(g_x_vars, ps)
            else:
                ys = func(*g_x_vars)
            ys = _as_tuple(ys)
            ys_data = tuple([None if y is None else y.array for y in ys])
            if xp is chainerx:
                ys_data = tuple([
                    None if y is None else y.as_grad_stopped()
                    for y in ys_data])

            if not self.is_immutable_params:
                for i, param in enumerate(params):
                    param.array = casted_data[j + i]

            return ys_data

        gx, = numerical_grad(
            g, (delta,), gys, eps=eps,
            detect_nondifferentiable=detect_nondifferentiable,
            center_outputs=y0_data, diff_atol=0, diff_rtol=self.rtol)

        return gx


def check_backward(
        func, x_data, y_grad, params=(),
        eps=1e-3, atol=1e-5, rtol=1e-4, no_grads=None, dtype=None,
        detect_nondifferentiable=False):
    """Test backward procedure of a given function.

    This function automatically checks the backward-process of a given function
    to ensure that the computed gradients are approximately correct.
    For example, assuming you've defined a :class:`~chainer.FunctionNode` class
    ``MyFunc``, that takes two arguments and returns one value, you can wrap
    it in a ordinary function and check its gradient computations as follows:

    .. code-block:: python

        def func(xs):
            y, = MyFunc().apply(xs)
            return y

        x1_data = xp.array(...)
        x2_data = xp.array(...)
        gy_data = xp.array(...)
        check_backward(func, (x1_data, x2_data), gy_data)

    This function creates :class:`~chainer.Variable` objects with ``x_data``
    and calls ``func`` with the :class:`~chainer.Variable`\\ s to get its
    result as :class:`~chainer.Variable`.
    Then, it sets ``y_grad`` array to ``grad`` attribute of the result and
    calls ``backward`` method to get gradients of the inputs.
    To check correctness of the gradients, the function calls
    :func:`numerical_grad` to calculate numerically the gradients and compares
    the types of gradients with :func:`chainer.testing.assert_allclose`.

    To reduce computational time, it uses directional derivative along a
    random vector. A function
    :math:`g: \\mathbb{R} \\rightarrow \\mathbb{R}^n` is defined as
    :math:`g(\\delta) = f(x + \\delta r)`, where
    :math:`\\delta \\in \\mathbb{R}`, :math:`r \\in \\mathbb{R}^n`
    is a random vector
    and :math:`f` is a function which you want to test.
    Its gradient is

    .. math::
       g'(\\delta) = f'(x + \\delta r) \\cdot r.

    Therefore, :math:`g'(0) = f'(x) \\cdot r`.
    So we can check the correctness of back propagation of :math:`f` indirectly
    by comparing this equation with the gradient of :math:`g` numerically
    calculated and that of :math:`f` computed by backprop.
    If :math:`r` is chosen from uniform distribution, we can conclude with
    high probability that the gradient of :math:`f` itself is correct.

    If the function is non-differentiable with respect to some input objects,
    we can check its backprop to such objects by ``no_grads`` argument.
    ``gradient_check`` computes numerical backward to inputs that correspond to
    ``False`` in ``no_grads``. It also asserts that the backprop leaves
    gradients ``None`` for inputs that correspond to ``True`` in ``no_grads``.
    The default of ``no_grads`` argument is the tuple of truth values whether
    input objects (``x1_data`` or/and ``x2_data`` in this example) represent
    integer variables.

    You can simplify a test when ``MyFunc`` gets only one argument:

    .. code-block:: python

        check_backward(func, x1_data, gy_data)

    If ``MyFunc`` is a loss function which returns a zero-dimensional
    array, pass ``None`` to ``gy_data``. In this case, it sets ``1`` to
    ``grad`` attribute of the result:

    .. code-block:: python

        check_backward(my_loss_func,
                       (x1_data, x2_data), None)

    If ``MyFunc`` returns multiple outputs, pass all gradients for outputs
    as a tuple:

    .. code-block:: python

        gy1_data = xp.array(...)
        gy2_data = xp.array(...)
        check_backward(func, x1_data, (gy1_data, gy2_data))

    You can also test a :class:`~chainer.Link`.
    To check gradients of parameters of the link, set a tuple of the parameters
    to ``params`` arguments:

    .. code-block:: python

        check_backward(my_link, (x1_data, x2_data), gy_data,
                       (my_link.W, my_link.b))

    Note that ``params`` are not ``ndarray``\\ s,
    but :class:`~chainer.Variables`\\ s.

    Function objects are acceptable as ``func`` argument:

    .. code-block:: python

        check_backward(lambda x1, x2: f(x1, x2),
                       (x1_data, x2_data), gy_data)

    .. note::

       ``func`` is called many times to get numerical gradients for all inputs.
       This function doesn't work correctly when ``func`` behaves randomly as
       it gets different gradients.


    Args:
        func (callable): A function which gets :class:`~chainer.Variable`\\ s
            and returns :class:`~chainer.Variable`\\ s. ``func`` must returns
            a tuple of :class:`~chainer.Variable`\\ s or one
            :class:`~chainer.Variable`. You can use a
            :class:`~chainer.Function`, :class:`~chainer.FunctionNode` or a
            :class:`~chainer.Link` object or any other function satisfying the
            condition.
        x_data (ndarray or tuple of ndarrays): A set of ``ndarray``\\ s to be
            passed to ``func``. If ``x_data`` is one ``ndarray`` object, it is
            treated as ``(x_data,)``.
        y_grad (ndarray or tuple of ndarrays or None):
            A set of ``ndarray``\\ s representing gradients of return-values of
            ``func``. If ``y_grad`` is one ``ndarray`` object, it is
            treated as ``(y_grad,)``. If ``func`` is a loss-function,
            ``y_grad`` should be set to ``None``.
        params (~chainer.Variable or tuple of ~chainder.Variable):
            A set of :class:`~chainer.Variable`\\ s whose gradients are
            checked. When ``func`` is a :class:`~chainer.Link` object,
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
        detect_nondifferentiable (bool):
            If ``True``, check for non-differentiable inputs is enabled.
            If ``func`` is non-differentiable at ``x_data``, ``check_backward``
            raises :class:`~chainer.gradient_check.NondifferentiableError`.

    .. seealso::
       :func:`numerical_grad`
    """
    _CheckBackward(
        func, x_data, y_grad, params, eps, atol, rtol, no_grads, dtype,
        detect_nondifferentiable, is_immutable_params=False
    ).run()


def _check_backward_with_params(
    # This function was introduced along with the `is_immutable_params`
    # argument to `_CheckBackward`.
    # It allows passing `params` as ndarrays instead of `Parameter`s and thus
    # depends less on the state of the parameter held by the caller.
    # It is required by the `LinkTestCase` to check ChainerX parameter
    # gradients, since those parameters cannot perturbed in-place for the
    # numerical gradients if passed as `Parameter`s as those requiring
    # gradients cannot be updated in-place.
        func, x_data, y_grad, params=(),
        eps=1e-3, atol=1e-5, rtol=1e-4, no_grads=None, dtype=None,
        detect_nondifferentiable=False):
    assert all(isinstance(p, chainer.get_array_types()) for p in params)
    _CheckBackward(
        func, x_data, y_grad, params, eps, atol, rtol, no_grads, dtype,
        detect_nondifferentiable, is_immutable_params=True
    ).run()


def check_double_backward(func, x_data, y_grad, x_grad_grad, params=(),
                          params_grad_grad=(), eps=1e-3, atol=1e-4, rtol=1e-3,
                          no_grads=None, dtype=None,
                          detect_nondifferentiable=False):
    """Test twice differentiation of a given procedure.

    This function automatically checks if the backward procedure of ``func``
    is correctly implemented for further differentiation. It first computes the
    gradient of ``func`` w.r.t. its inputs in the same way as
    :func:`~chainer.gradient_check.check_backward`. This function then further
    invokes the backward procedure against the gradient variables, starting
    from the initial gradient given by ``x_grad_grad``. It also computes the
    second gradient using :func:`~chainer.gradient_check.numerical_grad`. The
    resulting gradients are compared to confirm if the second-order gradients
    are approximately correct.

    Note that this function **DOES NOT** check if the first-order
    differentiation is correct; the numerical gradient assumes that the
    first-order gradient given by the usual :meth:`chainer.Variable.backward`
    is correct. The implementation of each differentiable function should be
    tested by :func:`~chainer.gradient_check.check_backward` first, and then
    should be tested by this function if neccessary.

    For the details of the arguments, see
    :func:`~chainer.gradient_check.check_backward`. The additional arguments
    ``x_grad_grad`` and ``params_grad_grad`` are (tuples of)
    :class:`~chainer.Variable` (s) that include the initial gradient
    corresponding to the first-order gradient of each input and parameter. Note
    that the default error tolerance ``atol`` and ``rtol`` are slightly larger
    than those of :func:`~chainer.gradient_check.check_backward` because the
    numerical gradients of the second order differentiation are less accurate
    than those of the first order gradients.

    """
    # Rename variables
    xs = x_data
    gys = y_grad
    ggxs = x_grad_grad
    ggparams = params_grad_grad
    no_gxs = no_grads
    del x_data
    del y_grad
    del x_grad_grad
    del params_grad_grad
    del no_grads

    xs = _as_tuple(xs)
    params = _as_tuple(params)
    gys = _as_tuple(gys)
    ggxs = _as_tuple(ggxs)
    ggparams = _as_tuple(ggparams)
    n_x = len(xs)

    first_order_no_gxs = [x.dtype.kind != 'f' for x in xs]

    def first_order_grad(*inputs):
        xs = inputs[:n_x]
        gys = inputs[n_x:]

        ys = _as_tuple(func(*xs))
        _check_outputs_and_grad_outputs(ys, gys)

        # Let all elements of y share the same creator.
        # See the comment in check_backward.
        y_backward = _apply_grad_setter_func(ys, gys)

        y_backward.backward(enable_double_backprop=True)

        gxs = []
        errors = []
        for i, (no_gx, x) in enumerate(six.moves.zip(first_order_no_gxs, xs)):
            if no_gx:
                if x.grad is not None:
                    errors.append(
                        '[{}]: Gradient was calculated while expected to not.'
                        .format(i))
            else:
                if x.grad is None:
                    gxs.append(None)
                else:
                    gxs.append(x.grad_var)

        if len(errors) > 0:
            f = six.StringIO()
            f.write('There are errors retrieving first-order gradients:\n')
            f.write('Inputs: {}\n'.format(utils._format_array_props(xs)))
            f.write('Skip: {}\n'.format(
                ', '.join(str(no_gx) for no_gx in first_order_no_gxs)))
            f.write('Errors:\n')
            for error in errors:
                f.write('{}\n'.format(error))
            raise RuntimeError(f.getvalue())

        return tuple(gxs + [p.grad_var for p in params])

    inputs = xs + gys
    grad_grad = ggxs + ggparams
    try:
        check_backward(first_order_grad, inputs, grad_grad, params=params,
                       eps=eps, atol=atol, rtol=rtol, no_grads=no_gxs,
                       dtype=dtype,
                       detect_nondifferentiable=detect_nondifferentiable)
    except AssertionError as e:
        f = six.StringIO()
        f.write('check_double_backward failed '
                '(eps={} atol={} rtol={})\n'.format(eps, atol, rtol))
        for i, x in enumerate(xs):
            f.write('input[{}]:\n'.format(i))
            f.write('{}\n'.format(x))
        for i, gy in enumerate(gys):
            f.write('grad_output[{}]:\n'.format(i))
            f.write('{}\n'.format(gy))
        for i, ggx in enumerate(ggxs):
            f.write('grad_grad_input[{}]:\n'.format(i))
            f.write('{}\n'.format(ggx))
        for i, ggp in enumerate(ggparams):
            f.write('grad_grad_param[{}]:\n'.format(i))
            f.write('{}\n'.format(ggp))
        f.write('\n')
        f.write(str(e))
        utils._raise_from(AssertionError, f.getvalue(), e)


class _GradientSetter(FunctionNode):
    def __init__(self, grad):
        self.grad = grad

    def forward(self, inputs):
        # output a dummy array.
        xp = backend.get_array_module(inputs[0])
        return xp.empty((0,), dtype=numpy.float32),

    def backward(self, inputs, grad_outputs):
        return self.grad


def _apply_grad_setter_func(ys, gys):
    # Applies the `_GradientSetter` function.
    # The gradient setter function accepts any number of upstream outputs as
    # its inputs, and returns a single output variable with dummy data.
    # This variable will be later backward()ed and during backprop, this
    # function returns the given gradients (`y_grad`) on its backward.
    assert len(ys) == len(gys)
    assert all(y is None or isinstance(y, chainer.Variable) for y in ys)
    assert all(gy is None or isinstance(gy, chainer.Variable) for gy in gys)
    # y is None => gy is None
    assert all(gy is None for y, gy in zip(ys, gys) if y is None)

    ys_ = [y for y in ys if y is not None]
    gys_ = [gy for y, gy in zip(ys, gys) if y is not None]
    assert len(ys_) == len(gys_)
    grad_setter = _GradientSetter(gys_)
    y, = grad_setter.apply(ys_)
    return y
