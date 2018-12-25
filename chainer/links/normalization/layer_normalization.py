from chainer.functions.normalization import layer_normalization
from chainer import link
from chainer import utils
from chainer import variable


class LayerNormalization(link.Link):

    """Layer normalization layer on outputs of linear functions.

    .. warning::

        This feature is experimental. The interface can change in the future.

    This link implements a "layer normalization" layer
    which normalizes the input units by statistics
    that are computed along the second axis,
    scales and shifts them.
    Parameter initialization will be deferred until
    the first forward data pass at which time the size will be determined.


    Args:
        size (int): Size of input units. If ``None``, parameter initialization
            will be deferred until the first forward data pass at which time
            the size will be determined.
        eps (float): Epsilon value for numerical stability of normalization.
        initial_gamma (~chainer.Initializer): Initializer for scaling vector.
            If ``None``, then the vector is filled by 1.
            If a scalar, the vector is filled by it.
            If ``numpy.ndarray``, the vector is set by it.
        initial_beta (~chainer.Initializer): Initializer for shifting vector.
            If ``None``, then the vector is filled by 0.
            If a scalar, the vector is filled by it.
            If ``numpy.ndarray``, the vector is set by it.

    Attributes:
        gamma (~chainer.Parameter): Scaling parameter.
        beta (~chainer.Parameter): Shifting parameter.
        eps (float): Epsilon value for numerical stability.

    See: `Layer Normalization <https://arxiv.org/abs/1607.06450>`_
    """

    def __init__(self, size=None, eps=1e-6, initial_gamma=None,
                 initial_beta=None):
        super(LayerNormalization, self).__init__()
        if initial_gamma is None:
            initial_gamma = 1
        if initial_beta is None:
            initial_beta = 0

        with self.init_scope():
            self.gamma = variable.Parameter(initial_gamma)
            self.beta = variable.Parameter(initial_beta)
            self.eps = eps

        if size is not None:
            self._initialize_params(size)

    def _initialize_params(self, size):
        self.gamma.initialize(size)
        self.beta.initialize(size)

    def forward(self, x):
        """Apply layer normalization to given input.

        Args:
            x (~chainer.Variable): Batch vectors.
                Shape of this value must be `(batch_size, unit_size)`,
                e.g., the output of :func:`~chainer.functions.linear`.

        Returns:
            ~chainer.Variable: Output of the layer normalization.

        """
        if self.gamma.array is None:
            in_size = utils.size_of_shape(x.shape[1:])
            self._initialize_params(in_size)

        return layer_normalization.layer_normalization(
            x, self.gamma, self.beta, self.eps)
