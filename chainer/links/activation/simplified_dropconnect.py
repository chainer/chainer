import numpy

from chainer.functions.noise import simplified_dropconnect
from chainer import initializers
from chainer import link
from chainer import variable


class SimplifiedDropconnect(link.Link):

    """Fully-connected layer with simplified dropconnect regularization.

    Notice:
    This implementation cannot be used for reproduction of the paper.
    There is a difference between the current implementation and the
    original one.
    The original version uses sampling with gaussian distribution before
    passing activation function, whereas the current implementation averages
    before activation.

    Args:
        in_size (int): Dimension of input vectors. If ``None``, parameter
            initialization will be deferred until the first forward data pass
            at which time the size will be determined.
        out_size (int): Dimension of output vectors.
        nobias (bool): If ``True``, then this link does not use the bias term.
        initialW (:ref:`initializer <initializer>`): Initializer to
            initialize the weight. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be 3.
        initial_bias (:ref:`initializer <initializer>`): Initializer to
            initialize the bias. If ``None``, the bias will be initialized to
            zero. When it is :class:`numpy.ndarray`, its ``ndim`` should be 2.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.

    .. seealso:: :func:`~chainer.functions.simplified_dropconnect`

    .. seealso::
        Li, W., Matthew Z., Sixin Z., Yann L., Rob F. (2013).
        Regularization of Neural Network using DropConnect.
        International Conference on Machine Learning.
        `URL <https://cs.nyu.edu/~wanli/dropc/>`_
    """

    def __init__(self, in_size, out_size, ratio=.5, nobias=False,
                 initialW=None, initial_bias=None):
        super(SimplifiedDropconnect, self).__init__()

        self.out_size = out_size
        self.ratio = ratio

        if initialW is None:
            initialW = initializers.HeNormal(1. / numpy.sqrt(2))

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)
            if in_size is not None:
                self._initialize_params(in_size)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = initializers.Constant(0)
                bias_initializer = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(bias_initializer, out_size)

    def _initialize_params(self, in_size):
        self.W.initialize((self.out_size, in_size))

    def __call__(self, x, train=True, mask=None, use_batchwise_mask=True):
        """Applies the simplified dropconnect layer.

        Args:
            x (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
                Batch of input vectors. Its first dimension ``n`` is assumed
                to be the *minibatch dimension*.
            train (bool):
                If ``True``, executes simplified dropconnect.
                Otherwise, simplified dropconnect link works as a linear unit.
            mask (None or chainer.Variable or numpy.ndarray or cupy.ndarray):
                If ``None``, randomized simplified dropconnect mask is
                generated. Otherwise, The mask must be ``(n, M, N)`` or
                ``(M, N)`` shaped array, and `use_batchwise_mask` is ignored.
                Main purpose of this option is debugging.
                `mask` array will be used as a dropconnect mask.
            use_batchwise_mask (bool):
                If ``True``, dropped connections depend on each sample in
                mini-batch.

        Returns:
            ~chainer.Variable: Output of the simplified dropconnect layer.

        """
        if self.W.data is None:
            self._initialize_params(x.size // len(x.data))
        if mask is not None and 'mask' not in self.__dict__:
            self.add_persistent('mask', mask)
        return simplified_dropconnect.simplified_dropconnect(
            x, self.W, self.b, self.ratio, train, mask, use_batchwise_mask)
