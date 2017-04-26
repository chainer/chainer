import numpy

from chainer import cuda
from chainer.functions.normalization import batch_normalization
from chainer import initializers
from chainer import link
from chainer import variable

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()


class BatchNormalization(link.Link):

    """Batch normalization layer on outputs of linear or convolution functions.

    This link wraps the :func:`~chainer.functions.batch_normalization` and
    :func:`~chainer.functions.fixed_batch_normalization` functions.

    It runs in three modes: training mode, fine-tuning mode, and testing mode.

    In training mode, it normalizes the input by *batch statistics*. It also
    maintains approximated population statistics by moving averages, which can
    be used for instant evaluation in testing mode.

    In fine-tuning mode, it accumulates the input to compute *population
    statistics*. In order to correctly compute the population statistics, a
    user must use this mode to feed mini-batches running through whole training
    dataset.

    In testing mode, it uses pre-computed population statistics to normalize
    the input variable. The population statistics is approximated if it is
    computed by training mode, or accurate if it is correctly computed by
    fine-tuning mode.

    Args:
        size (int or tuple of ints): Size (or shape) of channel
            dimensions.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability.
        dtype (numpy.dtype): Type to use in computing.
        use_gamma (bool): If ``True``, use scaling parameter. Otherwise, use
            unit(1) which makes no effect.
        use_beta (bool): If ``True``, use shifting parameter. Otherwise, use
            unit(0) which makes no effect.
        use_cudnn (bool): If ``True``, then this link uses cuDNN if available.

    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_

    .. seealso::
       :func:`~chainer.functions.batch_normalization`,
       :func:`~chainer.functions.fixed_batch_normalization`

    Attributes:
        gamma (~chainer.Variable): Scaling parameter.
        beta (~chainer.Variable): Shifting parameter.
        avg_mean (~chainer.Variable): Population mean.
        avg_var (~chainer.Variable): Population variance.
        N (int): Count of batches given for fine-tuning.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability. This value is added
            to the batch variances.
        use_cudnn (bool): If ``True``, then this link uses cuDNN if available.

    """

    def __init__(self, size, decay=0.9, eps=2e-5, dtype=numpy.float32,
                 use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None, use_cudnn=True):
        super(BatchNormalization, self).__init__()

        self.size_param = size
        self.dtype = dtype
        self.use_gamma = use_gamma
        self.use_beta = use_beta
        self.initial_gamma = initial_gamma
        self.initial_beta = initial_beta

        self.decay = decay
        self.eps = eps
        self.use_cudnn = use_cudnn

        self.do_init_params = True
        if not use_cudnn or dtype != numpy.float16:
            self.init_params()

    def init_params(self, x=None):
        """Initializes parameters such as gamma, beta, avg_mean and avg_var

        Parameters initilization was done in the :meth:`__init__`, but it
        has been separated from the :meth:`__init__`.

        When an argument `use_cudnn` of the :meth:`__init__` is `False` and
        cuDNN is never used, this method is called from the :meth:`__init__`,
        because data types of all parameters are always the same as data type
        of inputs that is specified by an argument `dtype` of the
        :meth:`__init__`.

        However, when the argument `use_cudnn` is `True` and cuDNN may be used,
        data type of parameters can not be determined until the
        :meth:`__call__` is called and the shape of inputs get known. In that
        case, this method is called from the :meth:`__call__` when it is first
        called.

        Args:
            x (Variable): Input variable.
        """
        self.do_init_params = False

        # [cuDNN]
        # if you want to use cudnn routines for batch normalization,
        # dtype of prameters like gamma, beta, etc. must be float32,
        # when dtype of input/output tensors are float16.
        dtype_param = self.dtype
        size_param = self.size_param
        initial_gamma = self.initial_gamma
        initial_beta = self.initial_beta
        if self.use_cudnn and cuda.cudnn_enabled and _cudnn_version >= 5000:
            xp = cuda.get_array_module(x)
            if xp is not numpy and (x.ndim == 2 or x.ndim == 4):
                if dtype_param == numpy.float16:
                    dtype_param = numpy.float32

        if self.use_gamma:
            self.add_param('gamma', size_param, dtype=dtype_param)
            if initial_gamma is None:
                initial_gamma = initializers.One()
            initializers.init_weight(self.gamma.data, initial_gamma)
        if self.use_beta:
            self.add_param('beta', size_param, dtype=dtype_param)
            if initial_beta is None:
                initial_beta = initializers.Zero()
            initializers.init_weight(self.beta.data, initial_beta)
        self.add_persistent('avg_mean',
                            numpy.zeros(size_param, dtype=dtype_param))
        self.add_persistent('avg_var',
                            numpy.zeros(size_param, dtype=dtype_param))
        self.add_persistent('N', 0)

        self.dtype_param = dtype_param

        # When this method is called from a method `__call__()`,
        # a method `to_gpu()` may be applied to this instance.
        # In that case, the method `to_gpu()` must be applied again
        # to make sure that all parameters initialized above are
        # also copied from CPU to GPU.
        if not self._cpu:
            self.to_cpu()
            self.to_gpu()

    def __call__(self, x, test=False, finetune=False):
        """Invokes the forward propagation of BatchNormalization.

        BatchNormalization accepts additional arguments, which controls three
        different running mode.

        Args:
            x (Variable): Input variable.
            test (bool): If ``True``, BatchNormalization runs in testing mode;
                it normalizes the input using pre-computed statistics.
            finetune (bool): If ``finetune`` is ``True`` and ``test`` is
                ``False``, BatchNormalization runs in fine-tuning mode; it
                accumulates the input array to compute population statistics
                for normalization, and normalizes the input using batch
                statistics.

        If ``test`` is ``False``, then BatchNormalization runs in training
        mode; it computes moving averages of mean and variance for evaluation
        during training, and normalizes the input using batch statistics.

        """
        if self.do_init_params:
            self.init_params(x)

        if hasattr(self, 'gamma'):
            gamma = self.gamma
        else:
            with cuda.get_device(self._device_id):
                gamma = variable.Variable(self.xp.ones(
                    self.avg_mean.shape, dtype=self.dtype_param),
                    volatile='auto')
        if hasattr(self, 'beta'):
            beta = self.beta
        else:
            with cuda.get_device(self._device_id):
                beta = variable.Variable(self.xp.zeros(
                    self.avg_mean.shape, dtype=self.dtype_param),
                    volatile='auto')

        if not test:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            func = batch_normalization.BatchNormalizationFunction(
                self.eps, self.avg_mean, self.avg_var, True, decay,
                self.use_cudnn)
            ret = func(x, gamma, beta)

            self.avg_mean[:] = func.running_mean
            self.avg_var[:] = func.running_var
        else:
            # Use running average statistics or fine-tuned statistics.
            mean = variable.Variable(self.avg_mean, volatile='auto')
            var = variable.Variable(self.avg_var, volatile='auto')
            ret = batch_normalization.fixed_batch_normalization(
                x, gamma, beta, mean, var, self.eps, self.use_cudnn)
        return ret

    def start_finetuning(self):
        """Resets the population count for collecting population statistics.

        This method can be skipped if it is the first time to use the
        fine-tuning mode. Otherwise, this method should be called before
        starting the fine-tuning mode again.

        """
        self.N = 0
