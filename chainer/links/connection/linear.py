import typing as tp  # NOQA

from chainer.functions.connection import linear
from chainer import initializers
from chainer import link
from chainer import types  # NOQA
from chainer import utils
from chainer import variable


class Linear(link.Link):

    """Linear layer (a.k.a.\\  fully-connected layer).

    This is a link that wraps the :func:`~chainer.functions.linear` function,
    and holds a weight matrix ``W`` and optionally a bias vector ``b`` as
    parameters.

    If ``initialW`` is left to the default value of ``None``, the weight matrix
    ``W`` is initialized with i.i.d. Gaussian samples, each of which has zero
    mean and deviation :math:`\\sqrt{1/\\text{in_size}}`. The bias vector ``b``
    is of size ``out_size``. If the ``initial_bias`` is to left the default
    value of ``None``, each element is initialized as zero.  If the ``nobias``
    argument is set to ``True``, then this link does not hold a bias vector.

    Args:
        in_size (int or None): Dimension of input vectors. If unspecified or
            ``None``, parameter initialization will be deferred until the
            first forward data pass at which time the size will be determined.
        out_size (int): Dimension of output vectors. If only one value is
            passed for ``in_size`` and ``out_size``, that value will be used
            for the ``out_size`` dimension.
        nobias (bool): If ``True``, then this function does not use the bias.
        initialW (:ref:`initializer <initializer>`): Initializer to initialize
            the weight. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be 2. If ``initialW`` is ``None``, then the
            weights are initialized with i.i.d. Gaussian samples, each of which
            has zero mean and deviation :math:`\\sqrt{1/\\text{in_size}}`.
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

        >>> x = np.array([[0, 1, 2, 3, 4]], np.float32)

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

    def __init__(
            self,
            in_size: tp.Optional[int],
            out_size: tp.Optional[int] = None,
            nobias: bool = False,
            initialW: tp.Optional[types.InitializerSpec] = None,
            initial_bias: tp.Optional[types.InitializerSpec] = None
    ) -> None:
        super(Linear, self).__init__()

        if out_size is None:
            in_size, out_size = None, in_size
        self.in_size = in_size
        self.out_size = out_size

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)  # type: variable.Variable  # NOQA
            if in_size is not None:
                self._initialize_params(in_size)

            if nobias:
                self.b = None  # type: tp.Optional[variable.Variable]
            else:
                if initial_bias is None:
                    initial_bias = 0
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_size)

    def _initialize_params(self, in_size: int) -> None:
        self.W.initialize((self.out_size, in_size))  # type: ignore

    @classmethod
    def from_params(cls, W, b=None, nobias=False):
        """Initialize a :class:`~chainer.links.Linear` with given parameters.

        This method uses ``W`` and optional ``b`` to initialize a linear layer.

        Args:
            W (:class:`~chainer.Variable` or :ref:`ndarray`):
                The weight parameter.
            b (:class:`~chainer.Variable`, :ref:`ndarray`, or ``None``):
                The bias parameter.
            nobias (bool): If ``True``, the argument of ``b`` is ignored
                in spite of whether it's given or not.
        """
        out_size, in_size = W.shape
        if b is not None:
            if out_size != b.size:
                raise ValueError('`out_size` does not match the size of `b`')
        link = cls(
            in_size, out_size, nobias,
            initialW=variable.as_array(W), initial_bias=variable.as_array(b))
        return link

    @property
    def printable_specs(self):
        specs = [
            ('in_size', self.in_size),
            ('out_size', self.out_size),
            ('nobias', self.b is None),
        ]
        for spec in specs:
            yield spec

    def forward(
            self,
            x: variable.Variable,
            n_batch_axes: int = 1
    ) -> variable.Variable:
        """Applies the linear layer.

        Args:
            x (~chainer.Variable): Batch of input vectors.
            n_batch_axes (int): The number of batch axes. The default is 1. The
                input variable is reshaped into
                (:math:`{\\rm n\\_batch\\_axes} + 1`)-dimensional tensor.
                This should be greater than 0.

        Returns:
            ~chainer.Variable: Output of the linear layer.

        """
        if self.W.array is None:
            in_size = utils.size_of_shape(x.shape[n_batch_axes:])
            self._initialize_params(in_size)
        return linear.linear(x, self.W, self.b, n_batch_axes=n_batch_axes)
