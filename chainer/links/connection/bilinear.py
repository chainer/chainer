import numpy

from chainer.backends import cuda
from chainer.functions.connection import bilinear
from chainer import initializers
from chainer import link
from chainer import variable


class Bilinear(link.Link):

    """Bilinear layer that performs tensor multiplication.

    Bilinear is a primitive link that wraps the
    :func:`~chainer.functions.bilinear` functions. It holds parameters ``W``,
    ``V1``, ``V2``, and ``b`` corresponding to the arguments of
    :func:`~chainer.functions.bilinear`.

    Args:
        left_size (int): Dimension of input vector :math:`e^1` (:math:`J`)
        right_size (int): Dimension of input vector :math:`e^2` (:math:`K`)
        out_size (int): Dimension of output vector :math:`y` (:math:`L`)
        nobias (bool): If ``True``, parameters ``V1``, ``V2``, and ``b`` are
            omitted.
        initialW (:ref:`initializer <initializer>`): Initializer to
            initialize the weight. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be 3.
        initial_bias (tuple of :ref:`initializer <initializer>`):
            Initial values of :math:`V^1`, :math:`V^2` and
            :math:`b`. The length of this argument must be 3.
            Each element of this tuple must have the shapes of
            ``(left_size, out_size)``, ``(right_size, out_size)``, and
            ``(out_size,)``, respectively if they are :class:`numpy.ndarray`.
            If ``None``, :math:`V^1` and :math:`V^2` are initialized
            by the default initializer and :math:`b` is set to :math:`0`.

    .. seealso:: See :func:`chainer.functions.bilinear` for details.

    Attributes:
        W (~chainer.Variable): Bilinear weight parameter.
        V1 (~chainer.Variable): Linear weight parameter for the first argument.
        V2 (~chainer.Variable): Linear weight parameter for the second
            argument.
        b (~chainer.Variable): Bias parameter.

    """

    def __init__(self, left_size, right_size, out_size, nobias=False,
                 initialW=None, initial_bias=None):
        super(Bilinear, self).__init__()
        self.in_sizes = (left_size, right_size)
        self.nobias = nobias

        # TODO(Kenta OONO): I do not know appropriate way of
        # initializing weights in tensor network.
        # This initialization is a modification of
        # that of Linear function.

        with self.init_scope():
            shape = (left_size, right_size, out_size)
            if isinstance(initialW, (numpy.ndarray, cuda.ndarray)):
                assert initialW.shape == shape
            self.W = variable.Parameter(
                initializers._get_initializer(initialW), shape)

            if not self.nobias:
                V1_shape = (left_size, out_size)
                V2_shape = (right_size, out_size)
                b_shape = (out_size,)
                if isinstance(initial_bias, tuple):
                    initialV1, initialV2, initialb = initial_bias
                    if isinstance(initialV1, (numpy.ndarray, cuda.ndarray)):
                        assert initialV1.shape == V1_shape
                    if isinstance(initialV2, (numpy.ndarray, cuda.ndarray)):
                        assert initialV2.shape == V2_shape
                    if isinstance(initialb, (numpy.ndarray, cuda.ndarray)):
                        assert initialb.shape == b_shape
                    initialV1 = initializers._get_initializer(initialV1)
                    initialV2 = initializers._get_initializer(initialV2)
                    initialb = initializers._get_initializer(initialb)
                elif initial_bias is None:
                    initialV1 = initializers._get_initializer(None)
                    initialV2 = initializers._get_initializer(None)
                    initialb = 0
                else:
                    raise ValueError('initial_bias must be tuple or None')

                self.V1 = variable.Parameter(initialV1, V1_shape)
                self.V2 = variable.Parameter(initialV2, V2_shape)
                self.b = variable.Parameter(initialb, b_shape)

    def forward(self, e1, e2):
        """Applies the bilinear function to inputs and the internal parameters.

        Args:
            e1 (~chainer.Variable): Left input.
            e2 (~chainer.Variable): Right input.

        Returns:
            ~chainer.Variable: Output variable.

        """
        if self.nobias:
            return bilinear.bilinear(e1, e2, self.W)
        else:
            return bilinear.bilinear(e1, e2, self.W, self.V1, self.V2, self.b)

    def zero_grads(self):
        # Left for backward compatibility
        self.zerograds()
