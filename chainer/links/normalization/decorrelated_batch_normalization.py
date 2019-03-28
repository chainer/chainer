import numpy

from chainer import configuration
from chainer import functions
from chainer import link
from chainer.utils import argument


class DecorrelatedBatchNormalization(link.Link):

    """Decorrelated batch normalization layer.

    This link wraps the
    :func:`~chainer.functions.decorrelated_batch_normalization` and
    :func:`~chainer.functions.fixed_decorrelated_batch_normalization`
    functions. It works on outputs of linear or convolution functions.

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
        groups (int): Number of groups to use for group whitening.
        decay (float): Decay rate of moving average
            which is used during training.
        eps (float): Epsilon value for numerical stability.
        dtype (numpy.dtype): Type to use in computing.

    See: `Decorrelated Batch Normalization <https://arxiv.org/abs/1804.08450>`_

    .. seealso::
       :func:`~chainer.functions.decorrelated_batch_normalization`,
       :func:`~chainer.functions.fixed_decorrelated_batch_normalization`

    Attributes:
        avg_mean (:ref:`ndarray`): Population mean.
        avg_projection (:ref:`ndarray`): Population
            projection.
        groups (int): Number of groups to use for group whitening.
        N (int): Count of batches given for fine-tuning.
        decay (float): Decay rate of moving average
            which is used during training.
        ~DecorrelatedBatchNormalization.eps (float): Epsilon value for
            numerical stability. This value is added to the batch variances.

    """

    def __init__(self, size, groups=16, decay=0.9, eps=2e-5,
                 dtype=numpy.float32):
        super(DecorrelatedBatchNormalization, self).__init__()
        self.avg_mean = numpy.zeros(size // groups, dtype=dtype)
        self.register_persistent('avg_mean')
        self.avg_projection = numpy.eye(size // groups, dtype=dtype)
        self.register_persistent('avg_projection')
        self.N = 0
        self.register_persistent('N')
        self.decay = decay
        self.eps = eps
        self.groups = groups

    def forward(self, x, **kwargs):
        """forward(self, x, *, finetune=False)

        Invokes the forward propagation of DecorrelatedBatchNormalization.

        In training mode, the DecorrelatedBatchNormalization computes moving
        averages of the mean and projection for evaluation during training,
        and normalizes the input using batch statistics.

        Args:
            x (:class:`~chainer.Variable`): Input variable.
            finetune (bool): If it is in the training mode and ``finetune`` is
                ``True``, DecorrelatedBatchNormalization runs in fine-tuning
                mode; it accumulates the input array to compute population
                statistics for normalization, and normalizes the input using
                batch statistics.

        """
        finetune, = argument.parse_kwargs(kwargs, ('finetune', False))

        if configuration.config.train:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            avg_mean = self.avg_mean
            avg_projection = self.avg_projection

            if configuration.config.in_recomputing:
                # Do not update statistics when extra forward computation is
                # called.
                if finetune:
                    self.N -= 1
                avg_mean = None
                avg_projection = None

            ret = functions.decorrelated_batch_normalization(
                x, groups=self.groups, eps=self.eps,
                running_mean=avg_mean, running_projection=avg_projection,
                decay=decay)
        else:
            # Use running average statistics or fine-tuned statistics.
            mean = self.avg_mean
            projection = self.avg_projection
            ret = functions.fixed_decorrelated_batch_normalization(
                x, mean, projection, groups=self.groups)
        return ret

    def start_finetuning(self):
        """Resets the population count for collecting population statistics.

        This method can be skipped if it is the first time to use the
        fine-tuning mode. Otherwise, this method should be called before
        starting the fine-tuning mode again.

        """
        self.N = 0
