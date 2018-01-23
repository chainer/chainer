import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import function
from chainer.functions.activation import log_softmax
from chainer.utils import type_check
from chainer import variable


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
            "only 'mean' and 'no' are valid for 'reduce', but '%s' is "
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


class SoftmaxCrossEntropy(function.Function):

    """Softmax activation followed by a cross entropy loss."""

    normalize = True
    y = None

    def __init__(self, normalize=True, cache_score=True, class_weight=None,
                 ignore_label=-1, reduce='mean'):
        self.normalize = normalize
        self.cache_score = cache_score
        _check_class_weight_option(class_weight)
        self.class_weight = class_weight
        self.ignore_label = ignore_label
        _check_reduce_option(reduce)
        self.reduce = reduce

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        x_type, t_type = in_types

        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype.kind == 'i',
            t_type.ndim == x_type.ndim - 1,

            x_type.shape[0] == t_type.shape[0],
            x_type.shape[2:] == t_type.shape[1:],
        )

    def forward_cpu(self, inputs):
        x, t = inputs
        if chainer.is_debug():
            _check_input_values(x, t, self.ignore_label)

        log_y = log_softmax._log_softmax(x)
        if self.cache_score:
            self.y = numpy.exp(log_y)
        if self.class_weight is not None:
            shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
            log_y *= _broadcast_to(self.class_weight.reshape(shape), x.shape)
        log_yd = numpy.rollaxis(log_y, 1)
        log_yd = log_yd.reshape(len(log_yd), -1)
        log_p = log_yd[numpy.maximum(t.ravel(), 0), numpy.arange(t.size)]

        log_p *= (t.ravel() != self.ignore_label)
        if self.reduce == 'mean':
            # deal with the case where the SoftmaxCrossEntropy is
            # unpickled from the old version
            if self.normalize:
                count = (t != self.ignore_label).sum()
            else:
                count = len(x)
            self._coeff = 1.0 / max(count, 1)

            y = log_p.sum(keepdims=True) * (-self._coeff)
            return y.reshape(()),
        else:
            return -log_p.reshape(t.shape),

    def forward_gpu(self, inputs):
        cupy = cuda.cupy
        x, t = inputs
        if chainer.is_debug():
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
        if self.class_weight is not None:
            shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
            log_y *= cupy.broadcast_to(
                self.class_weight.reshape(shape), x.shape)
        if self.normalize:
            coeff = cupy.maximum(1, (t != self.ignore_label).sum())
        else:
            coeff = max(1, len(t))
        self._coeff = cupy.divide(1.0, coeff, dtype=x.dtype)

        log_y = cupy.rollaxis(log_y, 1, log_y.ndim)
        if self.reduce == 'mean':
            ret = cuda.reduce(
                'S t, raw T log_y, int32 n_channel, raw T coeff, '
                'S ignore_label',
                'T out',
                't == ignore_label ? T(0) : log_y[_j * n_channel + t]',
                'a + b', 'out = a * -coeff[0]', '0', 'crossent_fwd'
            )(t, log_y.reduced_view(), log_y.shape[-1],
              self._coeff, self.ignore_label)
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

    def backward_cpu(self, inputs, grad_outputs):
        x, t = inputs
        gloss = grad_outputs[0]
        if x.size == 0:
            return numpy.zeros(x.shape, dtype=x.dtype), None
        if self.y is not None:
            y = self.y.copy()
        else:
            y = log_softmax._log_softmax(x)
            numpy.exp(y, out=y)
        if y.ndim == 2:
            gx = y
            gx[numpy.arange(len(t)), numpy.maximum(t, 0)] -= 1
            if self.class_weight is not None:
                shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
                c = _broadcast_to(self.class_weight.reshape(shape), x.shape)
                c = c[numpy.arange(len(t)), numpy.maximum(t, 0)]
                gx *= _broadcast_to(numpy.expand_dims(c, 1), gx.shape)
            gx *= (t != self.ignore_label).reshape((len(t), 1))
        else:
            # in the case where y.ndim is higher than 2,
            # we think that a current implementation is inefficient
            # because it yields two provisional arrays for indexing.
            n_unit = t.size // len(t)
            gx = y.reshape(y.shape[0], y.shape[1], -1)
            fst_index = numpy.arange(t.size) // n_unit
            trd_index = numpy.arange(t.size) % n_unit
            gx[fst_index, numpy.maximum(t.ravel(), 0), trd_index] -= 1
            if self.class_weight is not None:
                shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
                c = _broadcast_to(self.class_weight.reshape(shape), x.shape)
                c = c.reshape(gx.shape)
                c = c[fst_index, numpy.maximum(t.ravel(), 0), trd_index]
                c = c.reshape(y.shape[0], 1, -1)
                gx *= _broadcast_to(c, gx.shape)
            gx *= (t != self.ignore_label).reshape((len(t), 1, -1))
            gx = gx.reshape(y.shape)
        if self.reduce == 'mean':
            gx *= gloss * self._coeff
        else:
            gx *= gloss[:, None]
        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        cupy = cuda.cupy
        x, t = inputs
        if x.size == 0:
            return cupy.zeros(x.shape, dtype=x.dtype), None
        if self.y is not None:
            y = self.y
        else:
            y = log_softmax._log_softmax(x)
            cupy.exp(y, out=y)
        gloss = grad_outputs[0]
        n_unit = t.size // len(t)
        if self.reduce == 'mean':
            coeff = gloss * self._coeff
        else:
            coeff = gloss[:, None, ...]

        if self.class_weight is None:
            gx = cuda.elementwise(
                'T y, S t, T coeff, S n_channel, S n_unit, S ignore_label',
                'T gx',
                '''
                    const int c = (i / n_unit % n_channel);
                    gx = t == ignore_label ? 0 : coeff * (y - (c == t));
                ''',
                'softmax_crossent_bwd')(
                    y, cupy.expand_dims(t, 1), coeff, x.shape[1],
                    n_unit, self.ignore_label)
        else:
            gx = cuda.elementwise(
                'T y, raw T w, S t, T coeff, S n_channel, S n_unit, '
                'S ignore_label',
                'T gx',
                '''
                    const int c = (i / n_unit % n_channel);
                    gx = t == ignore_label ? 0 : coeff * (y - (c == t)) * w[t];
                ''',
                'softmax_crossent_weight_bwd')(
                    y, self.class_weight, cupy.expand_dims(t, 1), coeff,
                    x.shape[1], n_unit, self.ignore_label)

        return gx, None


def _double_backward_softmax_cross_entropy(x, t, normalize, class_weight,
                                           ignore_label, reduce):
    if isinstance(t, variable.Variable):
        t = t.data

    _check_class_weight_option(class_weight)
    _check_reduce_option(reduce)
    if chainer.is_debug():
        _check_input_values(x, t, ignore_label)

    loss = -chainer.functions.log_softmax(x)

    if class_weight is not None:
        shape = [1 if d != 1 else -1 for d in six.moves.range(x.ndim)]
        class_weight = chainer.functions.broadcast_to(
            class_weight.reshape(shape), x.shape)
        loss = loss * class_weight

    in_use = (t != ignore_label).astype(x.dtype)

    loss = chainer.functions.rollaxis(loss, 1, loss.ndim)
    loss = chainer.functions.reshape(loss, (-1, loss.shape[-1]))

    # Replace ignore_label value with one valid for F.select_item below.
    t = t.clip(0, loss.shape[1] - 1)

    loss = chainer.functions.select_item(loss, t.ravel())
    loss = chainer.functions.reshape(loss, t.shape)

    loss = loss * in_use

    if reduce == 'mean':
        if normalize:
            count = in_use.sum()
        else:
            count = len(x)
        count = max(count, 1.)
        loss = loss / count
        return chainer.functions.sum(loss)
    else:
        return loss


def softmax_cross_entropy(
        x, t, normalize=True, cache_score=True, class_weight=None,
        ignore_label=-1, reduce='mean', enable_double_backprop=False):
    """Computes cross entropy loss for pre-softmax activations.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variable holding a multidimensional array whose element indicates
            unnormalized log probability: the first axis of the variable
            represents the number of samples, and the second axis represents
            the number of classes. While this function computes a usual softmax
            cross entropy if the number of dimensions is equal to 2, it
            computes a cross entropy of the replicated softmax if the number of
            dimensions is greater than 2.
        t (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Variable holding a signed integer vector of ground truth
            labels. If ``t[i] == ignore_label``, corresponding ``x[i]`` is
            ignored.
        normalize (bool): If ``True``, this function normalizes the cross
            entropy loss across all instances. If ``False``, it only
            normalizes along a batch size.
        cache_score (bool): When it is ``True``, the function stores result
            of forward computation to use it on backward computation. It
            reduces computational cost though consumes more memory.
            If ``enable_double_backprop`` option is ``True``, this option
            is forcibly turned off and the function does not cache
            the intermediate value.
        class_weight (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
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

    Returns:
        ~chainer.Variable: A variable holding a scalar array of the cross
        entropy loss.  If ``reduce`` is ``'mean'``, it is a scalar array.
        If ``reduce`` is ``'no'``, the shape is same as that of ``x``.

    .. note::

       This function is differentiable only by ``x``.

    .. admonition:: Example

        >>> x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]]).astype('f')
        >>> x
        array([[-1.,  0.,  1.,  2.],
               [ 2.,  0.,  1., -1.]], dtype=float32)
        >>> t = np.array([3, 0]).astype('i')
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

    if enable_double_backprop:
        return _double_backward_softmax_cross_entropy(
            x, t, normalize, class_weight, ignore_label, reduce)
    else:
        return SoftmaxCrossEntropy(
            normalize, cache_score, class_weight, ignore_label, reduce)(x, t)
