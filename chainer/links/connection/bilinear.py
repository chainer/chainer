import math

import numpy

from chainer.functions.connection import bilinear
from chainer import link
from chainer import variable


class Bilinear(link.Link):

    """Bilinear link that wraps bilinear function.

    Bilinear is a primitive callable link that wraps the
    :func:`functions.bilinear` function. It holds parameters ``W``,
    ``V1``, ``V2``, and ``b`` corresponding to the arguments of
    :func:`functions.bilinear`.

    Args:
        left_size (int): Dimension of input vector :math:`e^1` (:math:`J`)
        right_size (int): Dimension of input vector :math:`e^2` (:math:`K`)
        out_size (int): Dimension of output vector :math:`y` (:math:`L`)
        nobias (bool): If ``True``, parameters ``V1``, ``V2``, and ``b`` are
            omitted.
        initialW (3-D Array): Initial value of :math:`W`.
            Shape of this argument must be
            ``(left_size, right_size, out_size)``. If ``None``,
            :math:`W` is initialized by centered Gaussian distribution properly
            scaled according to the dimension of inputs and outputs.
        initial_bias (tuple): Intial values of :math:`V^1`, :math:`V^2`
            and :math:`b`. The length this argument must be 3.
            Each element of this tuple must have the shapes of
            ``(left_size, output_size)``, ``(right_size, output_size)``,
            and ``(output_size,)``, respectively. If ``None``, :math:`V^1`
            and :math:`V^2` is initialized by scaled centered Gaussian
            distributions and :math:`b` is set to :math:`0`.

    .. seealso:: See :func:`functions.bilinear` for details.

    """
    def __init__(self, left_size, right_size, out_size, nobias=False,
                 initialW=None, initial_bias=None):
        super(Bilinear, self).__init__()
        self.in_sizes = (left_size, right_size)
        self.nobias = nobias

        if initialW is not None:
            assert initialW.shape == (left_size, right_size, out_size)
        else:
            # TODO(Kenta OONO): I do not know appropriate way of
            # initializing weights in tensor network.
            # This initialization is a modification of
            # that of Linear function.
            in_size = left_size * right_size * out_size
            initialW = numpy.random.normal(
                0, math.sqrt(1. / in_size), (left_size, right_size, out_size)
            ).astype(numpy.float32)
        self.params['W'] = variable.Variable(initialW)

        if not self.nobias:
            if initial_bias is not None:
                V1, V2, b = initial_bias
                assert V1.shape == (left_size, out_size)
                assert V2.shape == (right_size, out_size)
                assert b.shape == (out_size,)
            else:
                V1 = numpy.random.normal(
                    0, math.sqrt(1. / left_size), (left_size, out_size)
                ).astype(numpy.float32)
                V2 = numpy.random.normal(
                    0, math.sqrt(1. / right_size), (right_size, out_size)
                ).astype(numpy.float32)
                b = numpy.zeros(out_size, dtype=numpy.float32)
            self.params['V1'] = variable.Variable(V1)
            self.params['V2'] = variable.Variable(V2)
            self.params['b'] = variable.Variable(b)

    def __call__(self, e1, e2):
        if self.nobias:
            return bilinear.bilinear(e1, e2, self.params['W'])
        else:
            return bilinear.bilinear(
                e1, e2, self.params['W'], self.params['V1'],
                self.params['V2'], self.params['b'])
