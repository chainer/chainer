import chainer
from chainer import backend
from chainer import function_node
from chainer.utils import type_check


class Triplet(function_node.FunctionNode):

    """Triplet loss function."""

    def __init__(self, margin, reduce='mean'):
        if margin <= 0:
            raise ValueError('margin should be positive value.')
        self.margin = margin

        if reduce not in ('mean', 'no'):
            raise ValueError(
                'only \'mean\' and \'no\' are valid for \'reduce\', but '
                '\'%s\' is given' % reduce)
        self.reduce = reduce

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('anchor', 'positive', 'negative'))

        type_check.expect(
            in_types[0].dtype.kind == 'f',
            in_types[0].dtype == in_types[1].dtype,
            in_types[0].dtype == in_types[2].dtype,
            in_types[0].shape == in_types[1].shape,
            in_types[0].shape == in_types[2].shape,
            in_types[0].shape[0] > 0
        )

    def forward(self, inputs):
        xp = backend.get_array_module(*inputs)

        anchor, positive, negative = inputs

        dist = xp.sum(
            (anchor - positive) ** 2 - (anchor - negative) ** 2,
            axis=1) + self.margin
        self.dist_hinge = xp.maximum(dist, 0)
        if self.reduce == 'mean':
            N = anchor.shape[0]
            loss = xp.sum(self.dist_hinge) / N
        else:
            loss = self.dist_hinge

        self.retain_inputs((0, 1, 2))
        return xp.array(loss, dtype=anchor.dtype),

    def backward(self, indexes, grad_outputs):
        anchor, positive, negative = self.get_retained_inputs()

        N = anchor.shape[0]
        x_dim = anchor.shape[1]

        xp = backend.get_array_module(anchor)
        tmp = xp.repeat(self.dist_hinge[:, None], x_dim, axis=1)
        mask = xp.array(tmp > 0, dtype=anchor.dtype)

        gy, = grad_outputs
        if self.reduce == 'mean':
            g = gy / N
        else:
            g = gy[:, None]

        tmp = 2 * chainer.functions.broadcast_to(g, mask.shape) * mask

        ret = []
        if 0 in indexes:
            ret.append(tmp * (negative - positive))
        if 1 in indexes:
            ret.append(tmp * (positive - anchor))
        if 2 in indexes:
            ret.append(tmp * (anchor - negative))

        return ret


def triplet(anchor, positive, negative, margin=0.2, reduce='mean'):
    """Computes triplet loss.

    It takes a triplet of variables as inputs, :math:`a`, :math:`p` and
    :math:`n`: anchor, positive example and negative example respectively.
    The triplet defines a relative similarity between samples.
    Let :math:`N` and :math:`K` denote mini-batch size and the dimension of
    input variables, respectively. The shape of all input variables should be
    :math:`(N, K)`.

    .. math::
        L(a, p, n) = \\frac{1}{N} \\left( \\sum_{i=1}^N \\max \\{d(a_i, p_i)
            - d(a_i, n_i) + {\\rm margin}, 0\\} \\right)

    where :math:`d(x_i, y_i) = \\| {\\bf x}_i - {\\bf y}_i \\|_2^2`.

    The output is a variable whose value depends on the value of
    the option ``reduce``. If it is ``'no'``, it holds the elementwise
    loss values. If it is ``'mean'``, this function takes a mean of
    loss values.

    Args:
        anchor (:class:`~chainer.Variable` or :ref:`ndarray`):
            The anchor example variable. The shape
            should be :math:`(N, K)`, where :math:`N` denotes the minibatch
            size, and :math:`K` denotes the dimension of the anchor.
        positive (:class:`~chainer.Variable` or :ref:`ndarray`):
            The positive example variable. The shape
            should be the same as anchor.
        negative (:class:`~chainer.Variable` or :ref:`ndarray`):
            The negative example variable. The shape
            should be the same as anchor.
        margin (float): A parameter for triplet loss. It should be a positive
            value.
        reduce (str): Reduction option. Its value must be either
            ``'mean'`` or ``'no'``. Otherwise, :class:`ValueError` is raised.

    Returns:
        ~chainer.Variable:
            A variable holding a scalar that is the loss value
            calculated by the above equation.
            If ``reduce`` is ``'no'``, the output variable holds array
            whose shape is same as one of (hence both of) input variables.
            If it is ``'mean'``, the output variable holds a scalar value.

    .. note::
        This cost can be used to train triplet networks. See `Learning \
        Fine-grained Image Similarity with Deep Ranking \
        <https://arxiv.org/abs/1404.4661>`_ for details.

    .. admonition:: Example

        >>> anchor = np.array([[-2.0, 3.0, 0.5], [5.0, 2.0, -0.5]]).\
astype(np.float32)
        >>> pos = np.array([[-2.1, 2.8, 0.5], [4.9, 2.0, -0.4]]).\
astype(np.float32)
        >>> neg = np.array([[-2.1, 2.7, 0.7], [4.9, 2.0, -0.7]]).\
astype(np.float32)
        >>> F.triplet(anchor, pos, neg)
        variable(0.14000003)
        >>> y = F.triplet(anchor, pos, neg, reduce='no')
        >>> y.shape
        (2,)
        >>> y.array
        array([0.11000005, 0.17      ], dtype=float32)
        >>> F.triplet(anchor, pos, neg, margin=0.5)  # harder penalty
        variable(0.44000003)

    """
    return Triplet(margin, reduce).apply((anchor, positive, negative))[0]
