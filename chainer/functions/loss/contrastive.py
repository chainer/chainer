import numpy

from chainer.backends import cuda
from chainer import function
from chainer.utils import type_check


class Contrastive(function.Function):

    """Contrastive loss function."""

    def __init__(self, margin, reduce='mean'):
        if margin <= 0:
            raise ValueError("margin should be positive value.")
        self.margin = margin

        if reduce not in ('mean', 'no'):
            raise ValueError(
                "only 'mean' and 'no' are valid for 'reduce', but '%s' is "
                'given' % reduce)
        self.reduce = reduce

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)

        x0_type, x1_type, y_type = in_types
        type_check.expect(
            x0_type.dtype == numpy.float32,
            x1_type.dtype == numpy.float32,
            y_type.dtype.kind == 'i',
            x0_type.shape == x1_type.shape,
            x1_type.shape[0] == y_type.shape[0],
            x1_type.shape[0] > 0,
            x0_type.ndim == 2,
            x1_type.ndim == 2,
            y_type.ndim == 1
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x0, x1, y = inputs

        self.diff = x0 - x1
        self.dist_sq = xp.sum(self.diff ** 2, axis=1)
        self.dist = xp.sqrt(self.dist_sq)
        self.mdist = self.margin - self.dist
        dist = xp.maximum(self.mdist, 0)
        loss = (y * self.dist_sq + (1 - y) * dist * dist) * .5
        if self.reduce == 'mean':
            loss = xp.sum(loss) / x0.shape[0]
        return xp.array(loss, dtype=xp.float32),

    def backward(self, inputs, gy):
        xp = cuda.get_array_module(*inputs)
        x0, x1, y = inputs

        x_dim = x0.shape[1]
        y = xp.repeat(y[:, None], x_dim, axis=1)
        if self.reduce == 'mean':
            alpha = gy[0] / y.shape[0]
        else:
            alpha = gy[0][:, None]
        dist = xp.repeat(self.dist[:, None], x_dim, axis=1)
        # avoid division by zero
        dist = xp.maximum(dist, 1e-8)
        # similar pair
        gx0 = alpha * y * self.diff
        # dissimilar pair
        mdist = xp.maximum(xp.repeat(self.mdist[:, None], x_dim, axis=1), 0)
        gx0 += alpha * (1 - y) * mdist * -(self.diff / dist)
        gx0 = gx0.astype(xp.float32)

        return gx0, -gx0, None


def contrastive(x0, x1, y, margin=1, reduce='mean'):
    """Computes contrastive loss.

    It takes a pair of samples and a label as inputs.
    The label is :math:`1` when those samples are similar,
    or :math:`0` when they are dissimilar.

    Let :math:`N` and :math:`K` denote mini-batch size and the dimension
    of input variables, respectively. The shape of both input variables
    ``x0`` and ``x1`` should be ``(N, K)``.
    The loss value of the :math:`n`-th sample pair :math:`L_n` is

    .. math::
        L_n = \\frac{1}{2} \\left( y_n d_n^2
        + (1 - y_n) \\max ({\\rm margin} - d_n, 0)^2 \\right)

    where :math:`d_n = \\| {\\bf x_0}_n - {\\bf x_1}_n \\|_2`,
    :math:`{\\bf x_0}_n` and :math:`{\\bf x_1}_n` are :math:`n`-th
    K-dimensional vectors of ``x0`` and ``x1``.

    The output is a variable whose value depends on the value of
    the option ``reduce``. If it is ``'no'``, it holds the elementwise
    loss values. If it is ``'mean'``, this function takes a mean of
    loss values.

    Args:
        x0 (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): The first input variable. The shape should be
            (N, K), where N denotes the mini-batch size, and K denotes the
            dimension of ``x0``.
        x1 (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): The second input variable. The shape should be
            the same as ``x0``.
        y (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Labels. All values should be 0 or 1. The shape
            should be ``(N,)``, where N denotes the mini-batch size.
        margin (float): A parameter for contrastive loss. It should be positive
            value.
        reduce (str): Reduction option. Its value must be either
            ``'mean'`` or ``'no'``. Otherwise, :class:`ValueError` is raised.

    Returns:
        ~chainer.Variable:
            A variable holding the loss value(s) calculated by the
            above equation.
            If ``reduce`` is ``'no'``, the output variable holds array
            whose shape is same as one of (hence both of) input variables.
            If it is ``'mean'``, the output variable holds a scalar value.

    .. note::
        This cost can be used to train siamese networks. See `Learning a
        Similarity Metric Discriminatively, with Application to Face
        Verification <http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf>`_
        for details.

    .. admonition:: Example

        >>> x0 = np.array([[-2.0, 3.0, 0.5], [5.0, 2.0, -0.5]]).astype('f')
        >>> x1 = np.array([[-1.0, 3.0, 1.0], [3.5, 0.5, -2.0]]).astype('f')
        >>> y = np.array([1, 0]).astype('i')
        >>> F.contrastive(x0, x1, y)
        variable(0.3125)
        >>> F.contrastive(x0, x1, y, margin=3.0)  # harder penalty
        variable(0.3528857)
        >>> z = F.contrastive(x0, x1, y, reduce='no')
        >>> z.shape
        (2,)
        >>> z.data
        array([0.625, 0.   ], dtype=float32)

    """
    return Contrastive(margin, reduce)(x0, x1, y)
