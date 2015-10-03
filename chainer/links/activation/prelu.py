import numpy

from chainer.functions.activation import prelu
from chainer import link
from chainer import variable


class PReLU(link.Link):
    """Parametric ReLU function with attached parameters.

    Args:
        shape (tuple of ints): Shape of the parameter array.
        init (float): Initial parameter value.

    See detail in paper: `Delving Deep into Rectifiers: Surpassing \
    Human-Level Performance on ImageNet Classification \
    <http://arxiv.org/abs/1502.01852>`_.

    """

    def __init__(self, shape=(), init=0.25):
        super(PReLU, self).__init__()
        self.params['W'] = variable.Variable(
            numpy.full(shape, init, dtype=numpy.float32))

    def __call__(self, x):
        return prelu.prelu(x, self.params['W'])
