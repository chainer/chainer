import functools
import operator

from chainer.functions.connection import linear
from chainer import initializers
from chainer import link
from chainer import variable


class Linear(link.Link):

    """Linear layer (a.k.a.\\  fully-connected layer).

    This is a link that wraps the :func:`~chainer.functions.linear` function,
    and holds a weight matrix ``W`` and optionally a bias vector ``b`` as
    parameters.

    The weight matrix ``W`` is initialized with i.i.d. Gaussian samples, each
    of which has zero mean and deviation :math:`\\sqrt{1/\\text{in_size}}`. The
    bias vector ``b`` is of size ``out_size``. Each element is initialized with
    the ``bias`` value. If ``nobias`` argument is set to ``True``, then this
    link does not hold a bias vector.

    Args:
        in_size (int or None): Dimension of input vectors. If ``None``,
            parameter initialization will be deferred until the first forward
            data pass at which time the size will be determined.
        out_size (int): Dimension of output vectors.
        nobias (bool): If ``True``, then this function does not use the bias.
        initialW (:ref:`initializer <initializer>`): Initializer to initialize
            the weight. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be 2.
        initial_bias (:ref:`initializer <initializer>`): Initializer to
            initialize the bias. If ``None``, the bias will be initialized to
            zero. When it is :class:`numpy.ndarray`, its ``ndim`` should be 1.
    .. seealso:: :func:`~chainer.functions.linear`

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.

    .. admonition:: Example

        There are several ways to make a Linear link.

        Define an input vector ``x`` as:

        >>> x = np.array([[0, 1, 2, 3, 4]], 'f')

        1. Give the first two arguments explicitly:

            Those numbers are considered as the input size and the output size.

            >>> l = L.Linear(5, 10)
            >>> y = l(x)
            >>> y.shape
            (1, 10)

        2. Omit ``in_size`` (give the output size only as the first argument)
           or fill it with ``None``:

            In this case, the size of second axis of ``x`` is used as the
            input size. So the below two cases are the same.

            >>> l = L.Linear(10)
            >>> y = l(x)
            >>> y.shape
            (1, 10)

            >>> l = L.Linear(None, 10)
            >>> y = l(x)
            >>> y.shape
            (1, 10)

            When you omit the first argument, you need to specify the other
            subsequent arguments from ``nobias`` as keyword arguments. So the
            below two cases are the same.

            >>> l = L.Linear(None, 10, False, None, 0)
            >>> y = l(x)
            >>> y.shape
            (1, 10)

            >>> l = L.Linear(10, nobias=False, initialW=None, initial_bias=0)
            >>> y = l(x)
            >>> y.shape
            (1, 10)

    """

    def __init__(self, in_size, out_size=None, nobias=False,
                 initialW=None, initial_bias=None):
        super(Linear, self).__init__()

        if out_size is None:
            in_size, out_size = None, in_size
        self.out_size = out_size

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)
            if in_size is not None:
                self._initialize_params(in_size)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_size)

    def _initialize_params(self, in_size):
        self.W.initialize((self.out_size, in_size))

    def __call__(self, x):
        """Applies the linear layer.

        Args:
            x (~chainer.Variable): Batch of input vectors.

        Returns:
            ~chainer.Variable: Output of the linear layer.

        """
        if self.W.data is None:
            in_size = functools.reduce(operator.mul, x.shape[1:], 1)
            self._initialize_params(in_size)
        return linear.linear(x, self.W, self.b)
