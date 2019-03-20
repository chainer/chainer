import warnings

import chainer
from chainer.functions.normalization import instance_normalization
from chainer import initializers
from chainer import link
from chainer import variable
from chainer.utils import argument


class InstanceNormalization(link.Link):

    """Instance normalization layer on outputs of linear or \
convolution functions.

    This link wraps the :func:`~chainer.functions.instance_normalization`.
    Instance normalization is very close to batch normalization but different
    in that this normalizes each samples in a mini-batch by
    its mean and standard deviation even if in testing mode.
    However, you can use average mean and variance in testing mode by set
    ``track_avg_stats`` as ``True``. Note that this normalization only
    works on inputs whose dimensions are greater than 2.

    This runs in three modes: training mode, fine-tuning mode, and testing mode
    if you set ``track_avg_stats`` ``True``. See details of the above modes in
    :class:`~chainer.links.BatchNormalization`.

    Args:
        size (int): Size (or shape) of channel dimensions.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability.
        dtype (numpy.dtype): Type to use in computing.
        use_gamma (bool): If ``True``, use scaling parameter. Otherwise, use
            unit(1) which makes no effect.
        use_beta (bool): If ``True``, use shifting parameter. Otherwise, use
            unit(0) which makes no effect.
        track_avg_stats (bool): If ``True``, moving statistics are used in
            testing mode. The default value is ``False``.
        initial_gamma: Initializer of the scaling parameter. The default value
            is ``1``.
        initial_beta: Initializer of the shifting parameter. The default value
            is ``0``.
        initial_avg_mean: Initializer of the moving average of population mean.
            The default value is ``0``.
        initial_avg_var: Initializer of the moving average of population
            variance. The default value is ``1``.

    See: `Instance Normalization: The Missing Ingredient for Fast Stylization
           <https://arxiv.org/abs/1607.08022>`_

    .. seealso::
       :func:`~chainer.functions.instance_normalization`,

    Attributes:
        gamma (:class:`~chainer.Variable`): Scaling parameter.
        beta (:class:`~chainer.Variable`): Shifting parameter.
        avg_mean (:ref:`ndarray`): Population mean.
        avg_var (:ref:`ndarray): Population variance.
        N (int): Count of batches given for fine-tuning.
        decay (float): Decay rate of moving average. It is used on training.
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
    avg_mean = None
    avg_var = None

    def __init__(self, size, decay=0.9, eps=2e-5, dtype=None,
                 use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None,
                 track_avg_stats=False,
                 initial_avg_mean=None, initial_avg_var=None):
        super(InstanceNormalization, self).__init__()
        self.size = size
        self._initial_avg_mean = initial_avg_mean
        self._initial_avg_var = initial_avg_var
        self.decay = decay
        self.eps = eps
        self.track_avg_stats = track_avg_stats
        self._dtype = chainer.get_dtype(dtype)

        with self.init_scope():
            if use_gamma:
                if initial_gamma is None:
                    initial_gamma = 1
                gamma_initializer = \
                    initializers._get_initializer(initial_gamma)
                gamma_initializer.dtype = self._dtype
                self.gamma = variable.Parameter(gamma_initializer, (size,))
            if use_beta:
                if initial_beta is None:
                    initial_beta = 0
                beta_initializer = initializers._get_initializer(initial_beta)
                beta_initializer.dtype = self._dtype
                self.beta = variable.Parameter(beta_initializer, (size,))
        self.N = 0
        self.register_persistent('N')
        if self.track_avg_stats:
            self.avg_mean = self._init_array(self._initial_avg_mean, 0, size)
            self._initial_avg_mean = None
            self.register_persistent('avg_mean')
            self.avg_var = self._init_array(self._initial_avg_var, 1, size)
            self._initial_avg_var = None
            self.register_persistent('avg_var')

    def _init_array(self, initializer, default_value, size):
        if initializer is None:
            initializer = default_value
        initializer = initializers._get_initializer(initializer)
        return initializers.generate_array(
            initializer, size, self.xp, dtype=self._dtype)

    def forward(self, x, **kwargs):
        """forward(self, x, finetune=False)

        Invokes the forward propagation of InstanceNormalization.

        In training mode and ``track_avg_stats`` is ``True``,
        InstanceNormalization computes moving averages of mean and variance
        for evaluation during training.

        .. warning::

            ``test`` argument is not supported anymore since v2.
            Instead, use ``chainer.using_config('train', False)``.
            See :func:`chainer.using_config`.

        Args:
            x (:class:`~chainer.Variable`):
                Input variable, mini-batch of instances.
            finetune (bool): Finetune is triggered only when
                ``track_avg_stats`` is ``True`` and it is in training mode.
                In fine-tuning mode, this accumulates the input array
                to compute population statistics for normalization,
                and normalizes the input using instance statistics.

        Returns:
            ~chainer.Variable: Output variable with the same shape of ``x``.

        """
        finetune, = argument.parse_kwargs(
            kwargs, ('finetune', False),
            test='test argument is not supported anymore. '
                 'Use chainer.using_config')

        if finetune and not self.track_avg_stats:
            warnings.warn(
                'Because `track_avg_stats` is ``False``,'
                'finetune is ineffective.',
                UserWarning
            )

        gamma = self.gamma
        if gamma is None:
            with chainer.using_device(self.device):
                gamma = self.xp.ones(self.size, dtype=x.dtype)
        beta = self.beta
        if beta is None:
            with chainer.using_device(self.device):
                beta = self.xp.zeros(self.size, dtype=x.dtype)

        avg_mean, avg_var = self.avg_mean, self.avg_var
        decay = self.decay
        if chainer.config.train:
            if self.track_avg_stats:
                if finetune:
                    self.N += 1
                    decay = 1. - 1. / self.N
            if chainer.config.in_recomputing:
                if finetune:
                    self.N -= 1
                avg_mean = None
                avg_var = None

        if not chainer.config.train and self.track_avg_stats:
            ret = instance_normalization.fixed_instance_normalization(
                x, gamma, beta, avg_mean, avg_var, self.eps
            )
        else:
            ret = instance_normalization.instance_normalization(
                x, gamma, beta, eps=self.eps, running_mean=avg_mean,
                running_var=avg_var, decay=decay
            )
        return ret

    def start_finetuning(self):
        """Resets the population count for collecting population statistics.

        This method can be skipped if it is the first time to use the
        fine-tuning mode. Otherwise, this method should be called before
        starting the fine-tuning mode again.

        """
        if self.track_avg_stats:
            warnings.warn(
                'Because `track_avg_stats` is ``False``,'
                'finetune is ineffective.',
                UserWarning
            )
        self.N = 0
