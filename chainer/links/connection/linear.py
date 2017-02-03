from chainer import cuda
from chainer import initializers
from chainer import link
from chainer.functions.connection import linear


class Linear(link.Link):

    """Linear layer (a.k.a. fully-connected layer).

    This is a link that wraps the :func:`~chainer.functions.linear` function,
    and holds a weight matrix ``W`` and optionally a bias vector ``b`` as
    parameters.

    The weight matrix ``W`` is initialized with i.i.d. Gaussian samples, each
    of which has zero mean and deviation :math:`\\sqrt{1/\\text{in_size}}`. The
    bias vector ``b`` is of size ``out_size``. Each element is initialized with
    the ``bias`` value. If ``nobias`` argument is set to True, then this link
    does not hold a bias vector.

    Args:
        in_size (int): Dimension of input vectors. If it's ``None`` or ommited,
            parameter initialization will be deferred until the first forward
            data pass at which time the size will be determined.
        out_size (int): Dimension of output vectors.
        bias (float): Initial bias value.
        nobias (bool): If ``True``, then this function does not use the bias.
        initialW (2-D array): Initial weight value. If ``None``, the default
            initializer is used to initialize the weight matrix.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.

    .. seealso:: :func:`~chainer.functions.linear`

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.

    .. admonition:: Example

        There are three ways to make a Linear link.

        Define an input vector ``x`` as below,

        >>> x = np.random.rand(1, 5).astype(np.float32)

        and then,

        1. Give both input and output sizes:

            >>> l = L.Linear(5, 10)
            >>> y = l(x)
            >>> y.shape
            (1, 10)

        2. Give the output size only:

            >>> l = L.Linear(10)
            >>> y = l(x)
            >>> y.shape
            (1, 10)

        3. If you specify other arguments other than ``in_size`` and
            ``out_size``, you need to give them as keyword auguments.

            >>> l = L.Linear(5, 10, wscale=2, nobias=True)
            >>> y = l(x)
            >>> y.shape
            (1, 10)

            >>> l = L.Linear(10, wscale=2, nobias=True)
            >>> y = l(x)
            >>> y.shape
            (1, 10)

    """

    def __init__(self, in_size, out_size=None, bias=0, nobias=False,
                 initialW=None, initial_bias=None):
        super(Linear, self).__init__()

        # For backward compatibility
        self.initialW = initialW
        self._W_initializer = initializers._get_initializer(initialW)

        if in_size is None:
            self.out_size = out_size
            self.add_uninitialized_param('W')
        elif out_size is None:
            self.out_size = in_size
            self.add_uninitialized_param('W')
        else:
            self.out_size = out_size
            self._initialize_params(in_size)

        if nobias:
            self.b = None
        else:
            if initial_bias is None:
                initial_bias = bias
            bias_initializer = initializers._get_initializer(initial_bias)
            self.add_param('b', self.out_size, initializer=bias_initializer)

    def _initialize_params(self, in_size):
        self.add_param('W', (self.out_size, in_size),
                       initializer=self._W_initializer)

    def __call__(self, x):
        """Applies the linear layer.

        Args:
            x (~chainer.Variable): Batch of input vectors.

        Returns:
            ~chainer.Variable: Output of the linear layer.

        """
        if self.has_uninitialized_params:
            with cuda.get_device(self._device_id):
                self._initialize_params(x.size // x.shape[0])
        return linear.linear(x, self.W, self.b)
