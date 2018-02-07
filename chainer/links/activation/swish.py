from chainer.functions.activation import swish
from chainer import link
from chainer import variable


class Swish(link.Link):

    """Swish activation function as a link.

    Args:
        shape (tuple of ints): Shape of the parameter variable :math:`\\beta`.
        init (float): Initial value of the parameter variable :math:`\\beta`.

    See the paper for details: `Searching for Activation Functions \
    <https://arxiv.org/abs/1710.05941>`_

    .. seealso:: :func:`chainer.functions.swish`

    Attributes:
        beta (~chainer.Parameter): Parameter variable :math:`\\beta`.

    """

    def __init__(self, shape=(), init=1.0):
        super(Swish, self).__init__()
        with self.init_scope():
            self.beta = variable.Parameter(init, shape)

    def __call__(self, x):
        """Applies the Swish activation function.

        Args:
            x (~chainer.Variable): Input variable.

        Returns:
            ~chainer.Variable: Output of the Swish activation function.

        """
        return swish.swish(x, self.beta)
