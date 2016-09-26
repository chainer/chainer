from chainer import function


class Identity(function.Function):

    """Identity function."""

    def check_type_forward(self, in_types):
        pass

    def forward(self, xs):
        return xs

    def backward(self, xs, gys):
        return gys


def identity(*inputs):
    """Just returns input variables.


    Args:
        inputs (tuple of chainer.Variables, :class:`numpy.ndarray`s
        or cupy.ndarrays):
            Input variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return Identity()(*inputs)
