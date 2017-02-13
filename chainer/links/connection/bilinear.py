import numpy

from chainer import cuda
from chainer.functions.connection import bilinear
from chainer import initializers
from chainer import link


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
        initialW (3-D numpy array): Initial value of :math:`W`.
            Shape of this argument must be
            ``(left_size, right_size, out_size)``. If ``None``,
            :math:`W` is initialized by centered Gaussian distribution properly
            scaled according to the dimension of inputs and outputs.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        initial_bias (tuple): Initial values of :math:`V^1`, :math:`V^2`
            and :math:`b`. The length this argument must be 3.
            Each element of this tuple must have the shapes of
            ``(left_size, output_size)``, ``(right_size, output_size)``,
            and ``(output_size,)``, respectively. If ``None``, :math:`V^1`
            and :math:`V^2` is initialized by scaled centered Gaussian
            distributions and :math:`b` is set to :math:`0`.
            May also be a tuple of callables that take ``numpy.ndarray`` or
            ``cupy.ndarray`` and edit its value.

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

        shape = (left_size, right_size, out_size)
        if isinstance(initialW, (numpy.ndarray, cuda.ndarray)):
            assert initialW.shape == shape
        self.add_param('W', shape,
                       initializer=initializers._get_initializer(initialW))

        if not self.nobias:
            V1_shape = (left_size, out_size)
            V2_shape = (right_size, out_size)
            b_shape = (out_size,)
            if isinstance(initial_bias, tuple):
                V1, V2, b = initial_bias
                if isinstance(V1, (numpy.ndarray, cuda.ndarray)):
                    assert V1.shape == V1_shape
                if isinstance(V2, (numpy.ndarray, cuda.ndarray)):
                    assert V2.shape == V2_shape
                if isinstance(b, (numpy.ndarray, cuda.ndarray)):
                    assert b.shape == b_shape
                V1 = initializers._get_initializer(V1)
                V2 = initializers._get_initializer(V2)
                b = initializers._get_initializer(b)
            elif initial_bias is None:
                V1 = V2 = initializers._get_initializer(None)
                b = initializers.Constant(0)
            else:
                raise ValueError('initial_bias must be tuple or None')

            self.add_param('V1', V1_shape, initializer=V1)
            self.add_param('V2', V2_shape, initializer=V2)
            self.add_param('b', b_shape, initializer=b)

    def __call__(self, e1, e2):
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
