import numpy
import six

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function_node
from chainer.functions.activation import log_softmax
from chainer.utils import type_check
from chainer import variable
import chainerx


def _broadcast_to(array, shape):
    if hasattr(numpy, 'broadcast_to'):
        return numpy.broadcast_to(array, shape)
    dummy = numpy.empty(shape, array.dtype)
    return numpy.broadcast_arrays(array, dummy)[0]


def _check_class_weight_option(class_weight):
    if class_weight is not None:
        if class_weight.ndim != 1:
            raise ValueError('class_weight.ndim should be 1')
        if class_weight.dtype.kind != 'f':
            raise ValueError('The dtype of class_weight should be \'f\'')
        if isinstance(class_weight, variable.Variable):
            raise ValueError('class_weight should be a numpy.ndarray or '
                             'cupy.ndarray, not a chainer.Variable')


def _check_reduce_option(reduce):
    if reduce not in ('mean', 'no'):
        raise ValueError(
            'only \'mean\' and \'no\' are valid for \'reduce\', but \'%s\' is '
            'given' % reduce)


def _check_input_values(x, t, ignore_label):
    # Extract the raw ndarray as Variable.__ge__ is not implemented.
    # We assume that t is already an ndarray.
    if isinstance(x, variable.Variable):
        x = x.data

    if not (((0 <= t) &
             (t < x.shape[1])) |
            (t == ignore_label)).all():
        msg = ('Each label `t` need to satisfy '
               '`0 <= t < x.shape[1] or t == %d`' % ignore_label)
        raise ValueError(msg)


def _reduction_dtype(x_dtype):
    # Returns the dtype for accumulation and output of reduction.
    # For float16 input, float32 is used.
    # Otherwise the same dtype as the input is used.
    if x_dtype == numpy.float16:
        return numpy.float32
    return x_dtype


class SoftmaxCrossEntropy(function_node.FunctionNode):

    """Softmax activation followed by a cross entropy loss."""

    normalize = True
    y = None

    # Coefficient of normalization. Only used if reduce='mean'.
    _coeff = None
    soft_target = False
    eps = 1e-7

    def __init__(self, normalize=True, cache_score=True, class_weight=None,
                 ignore_label=-1, reduce='mean',
                 soft_target_loss='cross-entropy'):
        self.normalize = normalize
        self.cache_score = cache_score
        _check_class_weight_option(class_weight)
        self.class_weight = class_weight
        self.ignore_label = ignore_label
        _check_reduce_option(reduce)
        self.reduce = reduce
        self.soft_target_loss = soft_target_loss

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x', 't'))
        x_type, t_type = in_types

        if t_type.dtype.kind == 'i':
            type_check.expect(
                x_type.dtype.kind == 'f',
                t_type.dtype.kind == 'i',
                t_type.ndim == x_type.ndim - 1,
                x_type.shape[0] == t_type.shape[0],
                x_type.shape[2:] == t_type.shape[1:],
            )
        else:
            # assume t is soft_target
            type_check.expect(
                x_type.dtype.kind == 'f',
                t_type.dtype.kind == 'f',
                x_type.shape == t_type.shape,
            )

    def _is_chainerx_supported(self, input_arrays):
        # Determines if the specified configuration of inputs and parameters
        # are supported in `forward_chainerx` implementation.
        # TODO(niboshi): Support these conditions.
        if self.class_weight is not None:
            return False
        if self.ignore_label != -1:
            return False

        x, t = input_arrays

        if x.ndim != 2:
            return False

        return True

    def forward_chainerx(self, inputs):
        if self.reduce == 'mean' and self.normalize:
            x, t = inputs
            n_classes = x.shape[1]
            score = chainerx.log_softmax(x, axis=1)
            mask = (t[:, chainerx.newaxis] == chainerx.arange(
                n_classes, dtype=t.dtype, device=x.device)).astype(score.dtype)
            y = (score * mask).sum() * (-1 / mask.sum())
            return y,

        x, t = inputs
        y = chainerx.softmax_cross_entropy(x, t)
        if self.reduce == 'mean':
            return y.mean(),
        return y,

    def forward_cpu(self, inputs):
        class_weight = backend.from_chx(self.class_weight)

        self.retain_inputs((0, 1))
        x, t = inputs
        if x.ndim == t.ndim and x.shape == t.shape:
            self.soft_target = True
        if chainer.is_debug() and not self.soft_target:
            _check_input_values(x, t, self.ignore_label)

        log_y = log_softmax._log_softmax(x)
        if self.cache_score:
            self.y = numpy.exp(log_y)

        if self.soft_target:
            return self._soft_target_loss(numpy, x, t, log_y)

        if class_weight is not None:
            shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
            log_y *= _broadcast_to(class_weight.reshape(shape), x.shape)
        log_yd = numpy.rollaxis(log_y, 1)
        log_yd = log_yd.reshape(len(log_yd), -1)
        t_valid = t != self.ignore_label
        t = t * t_valid
        log_p = log_yd[t.ravel(), numpy.arange(t.size)]

        log_p *= t_valid.ravel()
        if self.reduce == 'mean':
            if self.normalize:
                count = t_valid.sum()
            else:
                count = len(x)
            self._coeff = 1.0 / max(count, 1)

            # Perform reduction in a promoted dtype
            reduc_dtype = _reduction_dtype(x.dtype)
            y = log_p.sum(keepdims=True, dtype=reduc_dtype)
            y = y * (-self._coeff)
            y = y.astype(x.dtype, copy=False)
            return y.reshape(()),
        else:
            return -log_p.reshape(t.shape),

    def forward_gpu(self, inputs):
        class_weight = backend.from_chx(self.class_weight)

        self.retain_inputs((0, 1))
        x, t = inputs
        if x.ndim == t.ndim and x.shape == t.shape:
            self.soft_target = True
        cupy = cuda.cupy
        if chainer.is_debug() and not self.soft_target:
            _check_input_values(x, t, self.ignore_label)

        if x.size == 0:
            y = cupy.zeros(t.shape, dtype=x.dtype)
            if self.cache_score:
                self.y = y
            if self.reduce == 'mean':
                return y.sum(),
            else:
                return y,
        log_y = log_softmax._log_softmax(x)
        if self.cache_score:
            self.y = cupy.exp(log_y)

        if self.soft_target:
            return self._soft_target_loss(cupy, x, t, log_y)

        if class_weight is not None:
            shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
            log_y *= cupy.broadcast_to(class_weight.reshape(shape), x.shape)

        log_y = cupy.rollaxis(log_y, 1, log_y.ndim)

        if self.reduce == 'mean':
            # Reduction is performed in a promoted dtype
            reduc_dtype = _reduction_dtype(x.dtype)
            if self.normalize:
                count = (t != self.ignore_label).sum(dtype=reduc_dtype)
                count = cupy.maximum(1, count)
                coeff = 1. / count
            else:
                coeff = cupy.array(1. / max(1, len(t)), dtype=reduc_dtype)
            self._coeff = coeff

            ret = cuda.reduce(
                'S t, raw T log_y, int32 n_channel, raw U coeff, '
                'S ignore_label',
                'U out',
                't == ignore_label ? T(0) : log_y[_j * n_channel + t]',
                'a + b', 'out = static_cast<U>(a * -coeff[0])', '0',
                'crossent_fwd'
            )(t, log_y.reduced_view(), log_y.shape[-1],
              self._coeff, self.ignore_label)
            ret = ret.astype(log_y.dtype, copy=False)
        else:
            ret = cuda.elementwise(
                'S t, raw T log_y, int32 n_channel, T ignore', 'T out',
                '''
                if (t == ignore) {
                  out = 0;
                } else {
                  out = -log_y[i * n_channel + t];
                }
                ''',
                'softmax_crossent_no_reduce_fwd'
            )(t, log_y.reduced_view(), log_y.shape[-1], self.ignore_label)
            ret = ret.reshape(t.shape)
        return ret,

    def _soft_target_loss(self, xp, x, t, log_y):
        if self.soft_target_loss == 'kl-divergence':
            ret = xp.sum(t * (xp.log(t + self.eps) - log_y), axis=1)
        else:
            ret = -xp.sum(t * log_y, axis=1)
        if self.reduce == 'mean':
            self._coeff = 1.0 / (x.size / x.shape[1])
            ret = ret.sum(keepdims=True) * self._coeff
            return ret.reshape(()),
        else:
            return ret,

    def backward(self, input_indexes, grad_outputs):
        func_grad = _SoftmaxCrossEntropyGrad_NoDoubleBackprop(
            self.ignore_label, self.class_weight, self.y, self._coeff,
            self.soft_target)
        inputs = self.get_retained_inputs()
        return func_grad.apply(inputs + grad_outputs) + (None,)


class _SoftmaxCrossEntropyGrad_NoDoubleBackprop(function_node.FunctionNode):
    # A backward implementation which does not support double-backprop.

    def __init__(self, ignore_label, class_weight, y, coeff, soft_target):
        self.ignore_label = ignore_label
        self.class_weight = class_weight
        self.y = y
        self.coeff = coeff
        self.soft_target = soft_target

    def forward_cpu(self, inputs_and_grad_outputs):
        x, t, gloss = inputs_and_grad_outputs
        if x.size == 0:
            return numpy.zeros(x.shape, dtype=x.dtype), None
        if self.y is not None:
            y = self.y.copy()
        else:
            y = log_softmax._log_softmax(x)
            numpy.exp(y, out=y)
        t_valid = t != self.ignore_label
        t = t * t_valid
        if self.soft_target:
            gx = y - t
        elif y.ndim == 2:
            gx = y
            gx[numpy.arange(len(t)), t] -= 1
            if self.class_weight is not None:
                shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
                c = _broadcast_to(self.class_weight.reshape(shape), x.shape)
                c = c[numpy.arange(len(t)), t]
                gx *= _broadcast_to(numpy.expand_dims(c, 1), gx.shape)
            gx *= t_valid.reshape((len(t), 1))
        else:
            # in the case where y.ndim is higher than 2,
            # we think that a current implementation is inefficient
            # because it yields two provisional arrays for indexing.
            n_unit = t.size // len(t)
            gx = y.reshape(y.shape[0], y.shape[1], -1)
            fst_index = numpy.arange(t.size) // n_unit
            trd_index = numpy.arange(t.size) % n_unit
            gx[fst_index, t.ravel(), trd_index] -= 1
            if self.class_weight is not None:
                shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
                c = _broadcast_to(self.class_weight.reshape(shape), x.shape)
                c = c.reshape(gx.shape)
                c = c[fst_index, t.ravel(), trd_index]
                c = c.reshape(y.shape[0], 1, -1)
                gx *= _broadcast_to(c, gx.shape)
            gx *= t_valid.reshape((len(t), 1, -1))
            gx = gx.reshape(y.shape)
        if self.coeff is not None:
            gx *= gloss * self.coeff
        else:
            gx *= gloss[:, None]
        return gx,

    def forward_gpu(self, inputs_and_grad_outputs):
        class_weight = cuda.to_gpu(self.class_weight)

        cupy = cuda.cupy
        x, t, gloss = inputs_and_grad_outputs
        if x.size == 0:
            return cupy.zeros(x.shape, dtype=x.dtype), None
        if self.y is not None:
            y = self.y
        else:
            y = log_softmax._log_softmax(x)
            cupy.exp(y, out=y)
        n_unit = t.size // len(t)
        if self.coeff is not None:
            coeff = self.coeff
        else:
            gloss = gloss[:, None, ...]
            coeff = cupy.array(1, dtype=gloss.dtype)  # dtype does not matter

        if self.soft_target:
            gx = gloss * coeff * (y - t)
        elif self.class_weight is None:
            gx = cuda.elementwise(
                'T y, S t, T gloss, U coeff, S n_channel, S n_unit, '
                'S ignore_label',
                'T gx',
                '''
                    const int c = (i / n_unit % n_channel);
                    if (t == ignore_label) {
                        gx = T(0);
                    } else {
                        gx = static_cast<T>(gloss * coeff * (y - (c == t)));
                    }
                ''',
                'softmax_crossent_bwd')(
                    y, cupy.expand_dims(t, 1), gloss, coeff, x.shape[1],
                    n_unit, self.ignore_label)
        else:
            gx = cuda.elementwise(
                'T y, raw T w, S t, T gloss, U coeff, '
                'S n_channel, S n_unit, S ignore_label',
                'T gx',
                '''
                    const int c = (i / n_unit % n_channel);
                    if (t == ignore_label) {
                        gx = T(0);
                    } else {
                        gx = static_cast<T>(
                            gloss * coeff * (y - (c == t)) * w[t]);
                    }
                ''',
                'softmax_crossent_weight_bwd')(
                    y, class_weight, cupy.expand_dims(t, 1), gloss, coeff,
                    x.shape[1], n_unit, self.ignore_label)

        return gx,

    def backward(self, input_indexes, grad_outputs):
        raise RuntimeError(
            'F.softmax_cross_entropy was called with '
            '\'enable_double_backprop=False\' argument, but double-backprop '
            'is actually being performed. Please specify '
            '\'enable_double_backprop=True\' explicitly.')


def _double_backward_softmax_cross_entropy(x, t, normalize, class_weight,
                                           ignore_label, reduce, is_chainerx):
    if isinstance(t, variable.Variable):
        t = t.data

    F = chainer.functions

    _check_class_weight_option(class_weight)
    _check_reduce_option(reduce)
    if chainer.is_debug():
        _check_input_values(x, t, ignore_label)

    loss = -chainer.functions.log_softmax(x)

    if class_weight is not None:
        shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
        class_weight = F.broadcast_to(class_weight.reshape(shape), x.shape)
        # TODO(niboshi): Remove this workaround after ChainerX supports
        # type promotion.
        if is_chainerx:
            class_weight = F.cast(class_weight, x.dtype)
        loss = loss * class_weight

    in_use = (t != ignore_label).astype(x.dtype)

    loss = F.rollaxis(loss, 1, loss.ndim)
    loss = F.reshape(loss, (-1, loss.shape[-1]))

    # Replace ignore_label value with one valid for F.select_item below.
    t = t.clip(0, loss.shape[1] - 1)

    loss = F.select_item(loss, t.ravel())
    loss = F.reshape(loss, t.shape)

    loss = loss * in_use

    if reduce == 'mean':
        reduc_dtype = _reduction_dtype(x.dtype)
        if normalize:
            # TODO(niboshi): Use in_use.sum(dtype=reduc_dtype) once chainerx
            # supports dtype argument.
            count = in_use.astype(reduc_dtype, copy=False).sum()
        else:
            count = len(x)
        count = max(count, 1.)

        if reduc_dtype == loss.dtype:
            loss = F.sum(loss / count)
        else:
            # Sum in a promoted dtype
            loss = F.cast(loss, reduc_dtype)
            loss = F.sum(loss / count)
            loss = F.cast(loss, x.dtype)

    return loss


def softmax_cross_entropy(
        x, t, normalize=True, cache_score=True, class_weight=None,
        ignore_label=-1, reduce='mean', enable_double_backprop=False,
        soft_target_loss='cross-entropy'):
    """Computes cross entropy loss for pre-softmax activations.

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Variable holding a multidimensional array whose element indicates
            unnormalized log probability: the first axis of the variable
            represents the number of samples, and the second axis represents
            the number of classes. While this function computes a usual softmax
            cross entropy if the number of dimensions is equal to 2, it
            computes a cross entropy of the replicated softmax if the number of
            dimensions is greater than 2.
        t (:class:`~chainer.Variable` or :ref:`ndarray`):
            Variable holding a signed integer vector of ground truth
            labels. If ``t[i] == ignore_label``, corresponding ``x[i]`` is
            ignored.
            When the dtype is float, this function treats ``t`` as an array
            holding probability distribution of labels, in other words, soft
            targets. In this case, the shape of ``t`` must be the same as the
            shape of ``x``. Note that the loss is calculated using cross
            entropy or KL divergence.
        normalize (bool): If ``True``, this function normalizes the cross
            entropy loss across all instances. If ``False``, it only
            normalizes along a batch size.
        cache_score (bool): When it is ``True``, the function stores result
            of forward computation to use it on backward computation. It
            reduces computational cost though consumes more memory.
            If ``enable_double_backprop`` option is ``True``, this option
            is forcibly turned off and the function does not cache
            the intermediate value.
        class_weight (:ref:`ndarray`):
            An array that contains constant weights that will be multiplied
            with the loss values along with the second dimension. The shape of
            this array should be ``(x.shape[1],)``. If this is not ``None``,
            each class weight ``class_weight[i]`` is actually multiplied to
            ``y[:, i]`` that is the corresponding log-softmax output of ``x``
            and has the same shape as ``x`` before calculating the actual loss
            value.
        ignore_label (int): Label value you want to ignore. Its default value
            is ``-1``. See description of the argument `t`.
        reduce (str): A string that determines whether to reduce the loss
            values. If it is ``'mean'``, it computes the sum of the individual
            cross entropy and normalize it according to ``normalize`` option.
            If it is ``'no'``, this function computes cross entropy for each
            instance and does not normalize it (``normalize`` option is
            ignored). In this case, the loss value of the ignored instance,
            which has ``ignore_label`` as its target value, is set to ``0``.
        enable_double_backprop (bool): If ``True``, this function uses
            implementation that supports higher order differentiation.
            If ``False``, it uses single-backprop implementation.
            This function use the single-backprop version because we expect
            it is faster. So, if you need second or higher derivatives,
            you need to turn it on explicitly.
        soft_target_loss (str): A string that determines what type of
            method is used to calculate soft target loss. If
            ``'cross-entropy'`` and ``'kl-divergence'``, cross-entropy and
            KL divergence are used for loss calculation.

    Returns:
        ~chainer.Variable: A variable holding a scalar array of the cross
        entropy loss.  If ``reduce`` is ``'mean'``, it is a scalar array.
        If ``reduce`` is ``'no'``, the shape is same as that of ``t``.

    .. note::

       This function is differentiable only by ``x``.

    .. admonition:: Example

        >>> x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]]).astype(np.float32)
        >>> x
        array([[-1.,  0.,  1.,  2.],
               [ 2.,  0.,  1., -1.]], dtype=float32)
        >>> t = np.array([3, 0]).astype(np.int32)
        >>> t
        array([3, 0], dtype=int32)
        >>> y = F.softmax_cross_entropy(x, t)
        >>> y
        variable(0.44018972)
        >>> log_softmax = -F.log_softmax(x)
        >>> expected_loss = np.mean([log_softmax[row, column].data \
for row, column in enumerate(t)])
        >>> y.array == expected_loss
        True

    """

    is_chainerx = (
        chainerx.is_available() and backend.get_array_module(x) is chainerx)

    if soft_target_loss not in ('cross-entropy', 'kl-divergence'):
        raise ValueError('soft_target_loss must be \'cross-entropy\' or '
                         '\'kl-divergence\'.')

    if is_chainerx or not enable_double_backprop:
        # Optimized implementation.
        # For non-ChainerX, forward and backward are supported but
        # double-backprop is not supported.
        # For ChainerX, even forward is supported for only specific
        # configuration of inputs and parameters, which is tested with
        # `SoftmaxCrossEntropy._is_chainerx_supported()`.
        func = SoftmaxCrossEntropy(
            normalize, cache_score, class_weight, ignore_label, reduce,
            soft_target_loss)

        if not is_chainerx or func._is_chainerx_supported((x, t)):
            loss, = func.apply((x, t))
            return loss

    # Generic double-backprop-enabled but unoptimized implementation
    return _double_backward_softmax_cross_entropy(
        x, t, normalize, class_weight, ignore_label, reduce, is_chainerx)
