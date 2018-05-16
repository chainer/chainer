from chainer.functions.activation import prelu
from chainer import link
from chainer import variable


class PReLU(link.Link):

    """Parametric ReLU function as a link.

    Args:
        shape (tuple of ints): Shape of the parameter array.
        init (float): Initial parameter value.

    See the paper for details: `Delving Deep into Rectifiers: Surpassing \
    Human-Level Performance on ImageNet Classification \
    <https://arxiv.org/abs/1502.01852>`_.

    .. seealso:: :func:`chainer.functions.prelu`

    Attributes:
        W (~chainer.Parameter): Coefficient of parametric ReLU.

    """

    def __init__(self, shape=(), init=0.25):
        super(PReLU, self).__init__()
        with self.init_scope():
            self.W = variable.Parameter(init, shape)

    def __call__(self, x):
        """Applies the parametric ReLU activation function.

        Args:
            x (~chainer.Variable): Input variable.

        Returns:
            ~chainer.Variable: Output of the parametric ReLU function.

        """
        return prelu.prelu(x, self.W)
