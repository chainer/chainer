from chainer.functions.normalization import group_normalization
from chainer import link
from chainer import variable


class GroupNormalization(link.Link):
    """Group normalization layer on outputs of convolution functions.

    This link implements a "group normalization"
    which divides the channels into groups and computes within each group
    the mean and variance, then normalize by these statistics,
    scales and shifts them.
    Parameter initialization will be deferred until
    the first forward data pass at which time the size will be determined.

    Args:
        groups (int):
            The number of channel groups.
            This value must be a divisor of the number of channels.
        size (int): Size of input units. If ``None``, parameter initialization
            will be deferred until the first forward data pass at which time
            the size will be determined.
        eps (float): Epsilon value for numerical stability of normalization.
        initial_gamma (~chainer.Initializer): Initializer for
            scaling parameter.
            If ``None``, then the vector is filled by 1.
            If a scalar, the vector is filled by it.
            If ``numpy.ndarray``, the vector is set by it.
        initial_beta (~chainer.Initializer): Initializer for
            shifting parameter.
            If ``None``, then the vector is filled by 0.
            If a scalar, the vector is filled by it.
            If ``numpy.ndarray``, the vector is set by it.

    Attributes:
        groups (int): The number of channel groups.
        gamma (~chainer.Parameter): Scaling parameter.
        beta (~chainer.Parameter): Shifting parameter.
        ~GroupNormalization.eps (float): Epsilon value for numerical stability.

    See: `Group Normalization <https://arxiv.org/abs/1803.08494>`_
    """

    def __init__(self, groups, size=None, eps=1e-5, initial_gamma=None,
                 initial_beta=None):
        super(GroupNormalization, self).__init__()
        if initial_gamma is None:
            initial_gamma = 1
        if initial_beta is None:
            initial_beta = 0

        with self.init_scope():
            self.groups = groups
            self.gamma = variable.Parameter(initial_gamma)
            self.beta = variable.Parameter(initial_beta)
            self.eps = eps

        if size is not None:
            self._initialize_params(size)

    def _initialize_params(self, size):
        self.gamma.initialize(size)
        self.beta.initialize(size)

    def forward(self, x):
        """Apply group normalization to given input.

        Args:
            x (~chainer.Variable): Batch tensors.
                First dimension of this value must be the size of minibatch and
                second dimension must be the number of channels.
                Moreover, this value must have one or more following
                dimensions, such as height and width.

        Returns:
            ~chainer.Variable: Output of the group normalization.

        """
        if self.gamma.array is None:
            if x.ndim < 2:
                raise ValueError('Input dimension must be at least 2, '
                                 'including batch size dimension '
                                 '(first dimension).')
            channels = x.shape[1]
            self._initialize_params(channels)

        return group_normalization.group_normalization(
            x, self.groups, self.gamma, self.beta, self.eps)
