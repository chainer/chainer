from chainer.functions.activation import relu
from chainer.functions.activation import sigmoid
from chainer import link
from chainer.links.connection import linear


class Highway(link.Chain):

    """Highway module.

    In highway network, two gates are added to the ordinal non-linear
    transformation (:math:`H(x) = activate(W_h x + b_h)`).
    One gate is the transform gate :math:`T(x) = \\sigma(W_t x + b_t)`, and the
    other is the carry gate :math:`C(x)`.
    For simplicity, the author defined :math:`C = 1 - T`.
    Highway module returns :math:`y` defined as

    .. math::

        y = activate(W_h x + b_h) \\odot \\sigma(W_t x + b_t) +
        x \\odot(1 - \\sigma(W_t x + b_t))

    The output array has the same spatial size as the input. In order to
    satisfy this, :math:`W_h` and :math:`W_t` must be square matrices.

    Args:
        in_out_size (int): Dimension of input and output vectors.
        nobias (bool): If ``True``, then this function does not use the bias.
        activate: Activation function of plain array. :math:`tanh` is also
            available.
        init_Wh (:ref:`initializer <initializer>`): Initializer to
            initialize the weight. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be 2.
        init_bh (:ref:`initializer <initializer>`): Initializer to
            initialize the bias. If ``None``, the bias will be initialized to
            zero. When it is :class:`numpy.ndarray`, its ``ndim`` should be 1.
        init_Wt (:ref:`initializer <initializer>`): Initializer to
            initialize the weight. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be 2.
        init_bt (:ref:`initializer <initializer>`): Initializer to
            initialize the bias. If ``None``, the bias will be initialized to
            zero. When it is :class:`numpy.ndarray`, its ``ndim`` should be 1.
            Negative value is recommended by the author of the paper.
            (e.g. -1, -3, ...).

    See:
        `Highway Networks <https://arxiv.org/abs/1505.00387>`_.
    """

    def __init__(self, in_out_size, nobias=False, activate=relu.relu,
                 init_Wh=None, init_Wt=None, init_bh=None, init_bt=-1):
        super(Highway, self).__init__()
        self.activate = activate

        with self.init_scope():
            self.plain = linear.Linear(
                in_out_size, in_out_size, nobias=nobias,
                initialW=init_Wh, initial_bias=init_bh)
            self.transform = linear.Linear(
                in_out_size, in_out_size, nobias=nobias,
                initialW=init_Wt, initial_bias=init_bt)

    def forward(self, x):
        """Computes the output of the Highway module.

        Args:
            x (~chainer.Variable): Input variable.
        Returns:
            ~chainer.Variable: Output variable. Its array has the same spatial
            size and the same minibatch size as the input array.
        """
        out_plain = self.activate(self.plain(x))
        out_transform = sigmoid.sigmoid(self.transform(x))
        y = out_plain * out_transform + x * (1 - out_transform)
        return y
