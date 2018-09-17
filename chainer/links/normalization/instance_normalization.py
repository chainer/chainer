import warnings

import numpy

import chainer
from chainer.backends import cuda
from chainer import configuration
from chainer import functions
from chainer import initializers
from chainer import link
from chainer.utils import argument
from chainer import variable


class InstanceNormalization(link.Link):

    """Instance normalization layer on outputs of linear or convolution functions.

    This link wraps the :func:`~chainer.functions.instance_normalization` and
    :func:`~chainer.functions.fixed_instance_normalization` functions.

    It runs in three modes: training mode, fine-tuning mode, and testing mode.

    In training mode, it normalizes the input by *sample statistics*. It also
    maintains approximated population statistics by moving averages, which can
    be used for instant evaluation in testing mode. Training mode is enabled
    when ``chainer.config.train`` is set to ``True`` and :meth:`__call__`
    is invoked with ``finetune=False`` (the default is False).

    In fine-tuning mode, it accumulates the input to compute *population
    statistics*. In order to correctly compute the population statistics, a
    user must use this mode to feed mini-batches running through whole training
    dataset. Finetuning mode is enabled when ``chainer.config.train`` is set to
    ``True`` and :meth:`__call__` is invoked with ``finetune=True``.

    In testing mode, it uses pre-computed population statistics to normalize
    the input variable. The population statistics is approximated if it is
    computed by training mode, or accurate if it is correctly computed by
    fine-tuning mode. Testing mode is enabled when ``chainer.config.train``
    is set to ``False``.

    Args:
        size (int, tuple of ints, or None): Size (or shape) of channel
            dimensions.  If ``None``, the size will be determined from
            dimension(s) of the input batch during the first forward pass.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability.
        dtype (numpy.dtype): Type to use in computing.
        track_running_stats (bool): If ``False``, running statistics
            will not be managed.
        use_gamma (bool): If ``True``, use scaling parameter. Otherwise, use
            unit(1) which makes no effect.
        use_beta (bool): If ``True``, use shifting parameter. Otherwise, use
            unit(0) which makes no effect.
        initial_gamma: Initializer of the scaling parameter. The default value
            is ``1``.
        initial_beta: Initializer of the shifting parameter. The default value
            is ``0``.
        initial_avg_mean: Initializer of the moving average of population mean.
            The default value is ``0``.
        initial_avg_var: Initializer of the moving average of population
            variance. The default value is ``1``.

    .. note::

        From v5.0.0, the initial value of the population variance is changed to
        1. It does not change the behavior of training, but the resulting model
        may have a slightly different behavior on inference. To emulate the
        old behavior, pass ``initial_avg_var=0`` for training.

    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_

    .. seealso::
       :func:`~chainer.functions.instance_normalization`,
       :func:`~chainer.functions.fixed_instance_normalization`

    Attributes:
        gamma (~chainer.Variable): Scaling parameter.
        beta (~chainer.Variable): Shifting parameter.
        avg_mean (numpy.ndarray or cupy.ndarray): Population mean.
        avg_var (numpy.ndarray or cupy.ndarray): Population variance.
        N (int): Count of batches given for fine-tuning.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability. This value is added
            to the batch variances.

    """  # NOQA

    gamma = None
    beta = None
    avg_mean = None
    avg_var = None

    def __init__(self, size, decay=0.9, eps=2e-5, dtype=None,
                 track_running_stats=False,
                 use_gamma=False, use_beta=False,
                 initial_gamma=None, initial_beta=None,
                 initial_avg_mean=None, initial_avg_var=None):
        super(InstanceNormalization, self).__init__()
        self.N = 0
        self.register_persistent('N')
        self.decay = decay
        self.eps = eps
        self.axis = None
        self._dtype = chainer.get_dtype(dtype)
        self.track_running_stats = track_running_stats

        with self.init_scope():
            self._initialize_params(size, initial_avg_mean, initial_avg_var)
            if use_gamma:
                if initial_gamma is None:
                    initial_gamma = 1
                gamma_initializer = initializers._get_initializer(initial_gamma)
                self.gamma = variable.Parameter(gamma_initializer)
            if use_beta:
                if initial_beta is None:
                    initial_beta = 0
                beta_initializer = initializers._get_initializer(initial_beta)
                self.beta = variable.Parameter(beta_initializer)

    def _initialize_params(self, shape, initial_avg_mean, initial_avg_var):
        dtype = self._dtype
        self.avg_mean = _init_array(None, initial_avg_mean, shape, dtype)
        self.avg_var = _init_array(None, initial_avg_var, shape, dtype)
        self.register_persistent('avg_mean')
        self.register_persistent('avg_var')

    def _get_gamma_beta(self, dtype):
        gamma = self.gamma
        if gamma is None:
            with cuda.get_device_from_id(self._device_id):
                gamma = self.xp.ones(
                    self.avg_mean.shape, dtype=dtype)

        beta = self.beta
        if beta is None:
            with cuda.get_device_from_id(self._device_id):
                beta = self.xp.zeros(
                    self.avg_mean.shape, dtype=dtype)
        return gamma, beta

    def forward(self, x, **kwargs):
        finetune, = argument.parse_kwargs(
            kwargs, ('finetune', False),
            test='test argument is not supported anymore. '
                 'Use chainer.using_config')

        gamma, beta = self._get_gamma_beta(x.dtype)

        if self.track_running_stats and not chainer.config.train:
            mean = self.avg_mean
            var = self.avg_var
            ret = functions.fixed_instance_normalization(
                x, gamma, beta, mean, var, self.eps, axis=self.axis)
        else:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            ret = functions.instance_normalization(
                x, gamma, beta, eps=self.eps,
                track_running_stat=self.track_running_stats,
                running_mean=self.avg_mean, running_var=self.avg_var,
                decay=decay, axis=self.axis)

        return ret

    def start_finetuning(self):
        if not self.track_running_stats:
            warnings.warn(
                'As track_running_stats is ``False``'
                ' finetuning does not effect at all.',
                UserWarning
            )

        self.N = 0


def _init_array(initializer, default_value, size, dtype):
    if initializer is None:
        initializer = default_value
    initializer = initializers._get_initializer(initializer)
    return initializers.generate_array(initializer, size, numpy, dtype=dtype)
