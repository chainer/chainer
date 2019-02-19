import chainer
from chainer.functions.array import to_finite
from chainer import link


class ToFinite(link.Link):
    """To Finite layer

    This is a link to force NaN and infinite values to finite value. If input
    array has NaN and infinite values, these is replaced by learnable
    parameters.

    Args:
        axis (int): The first axis of input variable along which parameters is
            applied.
        ndim (int): Number of axis of input variable along which parameters is
            applied.

    .. seealso:: See :func:`~chainer.functions.to_finite` for details.

    Attributes:
        nan_x (~chainer.Parameter): parameter to replace NaN values in input
            variable
        posinf_x (~chainer.Parameter): parameter to replace positive infinite
            values in input variable
        neginf_x (~chainer.Parameter): parameter to replace negative infinite
            values in input variable
    """

    def __init__(self, axis=1, ndim=1):
        super(ToFinite, self).__init__()
        self.axis = axis
        self.ndim = ndim

        with self.init_scope():
            self.nan_x = chainer.Parameter(0)
            self.posinf_x = chainer.Parameter(10)
            self.neginf_x = chainer.Parameter(-10)

    def _initialize_params(self, in_size):
        self.nan_x.initialize(in_size)
        self.posinf_x.initialize(in_size)
        self.neginf_x.initialize(in_size)

    def forward(self, x):
        """Force input to finite values.

        Args:
            x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variables
        """
        if self.nan_x.array is None:
            in_size = x.shape[self.axis: self.axis+self.ndim]
            self._initialize_params(in_size)
        x = to_finite.to_finite(x, self.nan_x, self.posinf_x, self.neginf_x,
                                self.axis)
        return x
