import chainer
from chainer.functions.normalization import instance_normalization
from chainer import initializers
from chainer import link
from chainer import variable


class InstanceNormalization(link.Link):

    """Instance normalization layer on outputs of linear or convolution functions.

    This link wraps the :func:`~chainer.functions.instance_normalization`.
    Instance normalization is very close to batch normalization but different
    in that this normalizes each samples in a mini-batch by its mean and standard
    deviation even if in testing mode. Also note that this normalization only
    works on inputs whose dimensions are greater than 2.

    Args:
        size (int): Size (or shape) of channel dimensions.
        eps (float): Epsilon value for numerical stability.
        dtype (numpy.dtype): Type to use in computing.
        use_gamma (bool): If ``True``, use scaling parameter. Otherwise, use
            unit(1) which makes no effect.
        use_beta (bool): If ``True``, use shifting parameter. Otherwise, use
            unit(0) which makes no effect.

        initial_gamma: Initializer of the scaling parameter. The default value
            is ``1``.
        initial_beta: Initializer of the shifting parameter. The default value
            is ``0``.

    See: `Instance Normalization: The Missing Ingredient for Fast Stylization
           <https://arxiv.org/abs/1607.08022>`_

    .. seealso::
       :func:`~chainer.functions.instance_normalization`,

    Attributes:
        gamma (~chainer.Variable): Scaling parameter.
        beta (~chainer.Variable): Shifting parameter.
        eps (float): Epsilon value for numerical stability. This value is added
            to the batch variances.

    .. admonition:: Example

        >>> x = np.arange(24).reshape(2, 3, 2, 2).astype(np.float32) ** 2
        >>> x
        array([[[[  0.,   1.],
                 [  4.,   9.]],
                [[ 16.,  25.],
                 [ 36.,  49.]],
                [[ 64.,  81.],
                 [100., 121.]]],
               [[[144., 169.],
                 [196., 225.]],
                [[256., 289.],
                 [324., 361.]],
                [[400., 441.],
                 [484., 529.]]]], dtype=float32)
        >>> instance_norm = chainer.links.InstanceNormalization(3)
        >>> instance_norm(x)
        variable([[[[-0.9999992 , -0.71428514],
                    [ 0.14285703,  1.5714273 ]],
                   [[-1.2561833 , -0.52678657],
                    [ 0.36469838,  1.4182715 ]],
                   [[-1.2931336 , -0.4937419 ],
                    [ 0.39969584,  1.3871796 ]]],
                  [[[-1.3077965 , -0.48007718],
                    [ 0.41385964,  1.374014  ]],
                   [[-1.3156561 , -0.47261432],
                    [ 0.4215209 ,  1.3667495 ]],
                   [[-1.3205545 , -0.467913  ],
                    [ 0.42632073,  1.3621467 ]]]])
        >>> (x - x.mean(axis=0)) / np.sqrt(x.var(axis=0) + 2e-5)
        array([[[[-0.99999917, -0.71428514],
                 [ 0.14285703,  1.5714273 ]],
                [[-1.2561833 , -0.52678657],
                 [ 0.36469838,  1.4182714 ]],
                [[-1.2931336 , -0.49374193],
                 [ 0.39969584,  1.3871797 ]]],
               [[[-1.3077965 , -0.4800772 ],
                 [ 0.41385964,  1.374014  ]],
                [[-1.3156562 , -0.47261435],
                 [ 0.4215209 ,  1.3667495 ]],
                [[-1.3205545 , -0.467913  ],
                 [ 0.42632073,  1.3621467 ]]]], dtype=float32)

    """

    gamma = None
    beta = None

    def __init__(self, size, eps=2e-5, dtype=None,
                 use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None):
        super(InstanceNormalization, self).__init__()
        self.eps = eps
        self._dtype = chainer.get_dtype(dtype)

        with self.init_scope():
            if use_gamma:
                if initial_gamma is None:
                    initial_gamma = 1
                gamma_initializer = \
                    initializers._get_initializer(initial_gamma)
                gamma_initializer.dtype = self._dtype
                self.gamma = variable.Parameter(gamma_initializer)
            if use_beta:
                if initial_beta is None:
                    initial_beta = 0
                beta_initializer = initializers._get_initializer(initial_beta)
                beta_initializer.dtype = self._dtype
                self.beta = variable.Parameter(beta_initializer)

        if size is not None:
            self._initialize_params(size)

    def _initialize_params(self, shape):
        if self.gamma is not None:
            self.gamma.initialize(shape)
        if self.beta is not None:
            self.beta.initialize(shape)

    def _init_array(self, initializer, default_value, size):
        if initializer is None:
            initializer = default_value
        initializer = initializers._get_initializer(initializer)
        return initializers.generate_array(
            initializer, size, self.xp, dtype=self._dtype)

    def forward(self, x):
        """forward(self, x, finetune=False)

        Args:
            x (Variable): Input variable.

        """

        return instance_normalization.instance_normalization(
            x, self.gamma, self.beta, self.eps)
