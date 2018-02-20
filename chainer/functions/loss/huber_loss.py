import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


class HuberLoss(function_node.FunctionNode):

    def __init__(self, delta, reduce='sum_along_second_axis'):
        self.delta = delta

        if reduce not in ('sum_along_second_axis', 'no'):
            raise ValueError(
                "Only 'sum_along_second_axis' and 'no' are valid "
                "for 'reduce', but '%s' is given" % reduce)

        self.reduce = reduce

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        xp = cuda.get_array_module(*inputs)
        x0, x1 = inputs
        diff = x0 - x1
        y = xp.square(diff)
        mask = y > (self.delta ** 2)
        y -= mask * xp.square(abs(diff) - self.delta)
        y *= 0.5
        if self.reduce == 'sum_along_second_axis':
            return y.sum(axis=1),
        else:
            return y,

    def backward(self, indexes, grad_outputs):
        x0, x1 = self.get_retained_inputs()
        diff = x0 - x1
        gy, = grad_outputs

        xp = cuda.get_array_module(diff)
        mask = xp.abs(diff.array) <= self.delta

        gx = chainer.functions.where(
            mask, diff, self.delta * xp.sign(diff.array))

        if self.reduce == 'sum_along_second_axis':
            gy = gy.reshape(gy.shape + (1,) * (diff.ndim - 1))
        gx = chainer.functions.broadcast_to(gy, gx.shape) * gx
        return gx, -gx


def huber_loss(x, t, delta, reduce='sum_along_second_axis'):
    """Computes the Huber loss.

    The Huber loss is similar to the :func:`mean_squared_error` but is less
    sensitive to outliers in the data. It is defined as

    .. math::

        L_{\\delta}(a) = \\left \\{ \\begin{array}{cc}
        \\frac{1}{2} a^2 & {\\rm if~|a| \\leq \\delta} \\\\
        \\delta (|a| - \\frac{1}{2} \\delta) & {\\rm otherwise,}
        \\end{array} \\right.

    where :math:`a = x - t` is the difference between the input :math:`x`
    and the target :math:`t`.

    The loss is a variable whose value depends on the value of
    the option ``reduce``. If it is ``'no'``, it holds the elementwise
    loss values. If it is ``'sum_along_second_axis'``, loss values are
    summed up along the second axis (i.e. ``axis=1``).

    See: `Huber loss - Wikipedia <https://en.wikipedia.org/wiki/Huber_loss>`_.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable.
            The shape of ``x`` should be (:math:`N`, :math:`K`).
        t (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Target variable for regression.
            The shape of ``t`` should be (:math:`N`, :math:`K`).
        delta (float): Constant variable for Huber loss function
            as used in definition.
        reduce (str): Reduction option. Its value must be either
            ``'sum_along_second_axis'`` or ``'no'``. Otherwise,
            :class:`ValueError` is raised.

    Returns:
        ~chainer.Variable:
            A variable object holding a scalar array of the
            Huber loss :math:`L_{\\delta}`.
            If ``reduce`` is ``'no'``, the output variable holds array
            whose shape is same as one of (hence both of) input variables.
            If it is ``'sum_along_second_axis'``, the shape of the array
            is same as the input variables, except the second axis is removed.

    .. admonition:: Example

        Example without reduction, in which case the output ``y`` will have the
        same shape as the inputs ``x`` and ``t``.

        >>> import numpy as np
        >>> from chainer import functions as F
        >>> x = np.array([[-2.0, 3.0, 0.5], [5.0, 2.0, -0.5]]).\
astype(np.float32)
        >>> x.shape
        (2, 3)
        >>> t = np.array([[-2.0, 3.0, 0.0], [10.0, 2.0, -0.5]]).\
astype(np.float32)
        >>> t.shape
        (2, 3)
        >>> y = F.huber_loss(x, t, delta=1.0, reduce='no')
        >>> y.shape
        (2, 3)
        >>> y
        variable([[0.   , 0.   , 0.125],
                  [4.5  , 0.   , 0.   ]])

        Example with reduction along the second axis.

        >>> y = F.huber_loss(x, t, delta=1.0, reduce='sum_along_second_axis')
        >>> y.shape
        (2,)
        >>> y
        variable([0.125, 4.5  ])

    """
    return HuberLoss(delta=delta, reduce=reduce).apply((x, t))[0]
