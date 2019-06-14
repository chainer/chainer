from chainer.functions.activation import elu


def selu(x,
         alpha=1.6732632423543772848170429916717,
         scale=1.0507009873554804934193349852946):
    """Scaled Exponential Linear Unit function.

    For parameters :math:`\\alpha` and :math:`\\lambda`, it is expressed as

    .. math::
        f(x) = \\lambda \\left \\{ \\begin{array}{ll}
        x & {\\rm if}~ x \\ge 0 \\\\
        \\alpha (\\exp(x) - 1) & {\\rm if}~ x < 0,
        \\end{array} \\right.

    See: https://arxiv.org/abs/1706.02515

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        alpha (float): Parameter :math:`\\alpha`.
        scale (float): Parameter :math:`\\lambda`.

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    """
    return scale * elu.elu(x, alpha=alpha)
