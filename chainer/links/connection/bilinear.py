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
        initialW (callable): Initializer for :math:`W`.
            It should be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            If it is ``None``, the default initializer is used.
            If it is `numpy.ndarray`, the array is used as initial
            weight value.
            Shape of the array must be ``(left_size, right_size, out_size)``.
        initial_bias (tuple): Bias initializers.
            It should be a 3-tuple of callables that takes ``numpy.ndarray``
            or ``cupy.ndarray`` and edits its value.
            They initialize :math:`V^1`, :math:`V^2`,  and :math:`b`,
            respectively.
            If it is ``None``, the default initializer is used.
            If it is a tuple of `numpy.ndarray`, the arrays are used as initial
            bias value.
            Each element of this tuple must have the shapes of
            ``(left_size, output_size)``, ``(right_size, output_size)``,
            and ``(output_size,)``, respectively.

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
        super(Bilinear, self).__init__(W=(left_size, right_size, out_size))
        self.in_sizes = (left_size, right_size)
        self.nobias = nobias

        # TODO(Kenta OONO): I do not know appropriate way of
        # initializing weights in tensor network.
        # This initialization is a modification of
        # that of Linear function.

        if isinstance(initialW, (numpy.ndarray, cuda.ndarray)):
            assert initialW.shape == self.W.shape
        if initialW is None:
            initialW = initializers.HeNormal(1.0 / numpy.sqrt(2))
        initializers.init_weight(self.W.data, initialW)

        if not self.nobias:
            self.add_param('V1', (left_size, out_size))
            self.add_param('V2', (right_size, out_size))
            self.add_param('b', out_size)

            if initial_bias is None:
                V1 = initializers.HeNormal(1.0 / numpy.sqrt(2))
                V2 = initializers.HeNormal(1.0 / numpy.sqrt(2))
                b = initializers.Constant(0)
            else:
                if len(initial_bias) != 3:
                    raise ValueError('initial_bias must be tuple or None')
                V1, V2, b = initial_bias
                if isinstance(V1, (numpy.ndarray, cuda.ndarray)):
                    assert V1.shape == self.V1.shape
                if isinstance(V2, (numpy.ndarray, cuda.ndarray)):
                    assert V2.shape == self.V2.shape
                if isinstance(b, (numpy.ndarray, cuda.ndarray)):
                    assert b.shape == self.b.shape

            initializers.init_weight(self.V1.data, V1)
            initializers.init_weight(self.V2.data, V2)
            initializers.init_weight(self.b.data, b)

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
