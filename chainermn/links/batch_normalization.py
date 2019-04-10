import chainer
from chainer import cuda
from chainer.functions.normalization import batch_normalization
from chainer import initializers
from chainer import link
import chainer.utils
from chainer import variable
import numpy

from chainermn.functions.batch_normalization import \
    get_communication_backend
from chainermn.functions.batch_normalization import \
    MultiNodeBatchNormalizationFunction


class MultiNodeBatchNormalization(link.Link):

    """Batch normalization layer that can use the whole batch stats.

    When using chainer.link.BatchNormalization, batch mean and std are
    computed independently for the local batch in each worker. When local
    batch size is too small, training is unstable due to unreliable batch
    stats.

    In contrast, when using this MultiNodeBatchNormalization, workers
    communicate to conduct 'correct' batch normalization (e.g., obtaining
    mean and std for the whole global batch).

    This link works only with Chainer >= 2.0.0.

    Args:
        size (int or tuple of ints): Size (or shape) of channel
            dimensions.
        comm (ChainerMN communicator): communicator to share
            the batch stats.
        decay (float): Decay rate of moving average. It is used on training.
        eps (float): Epsilon value for numerical stability.
        dtype (numpy.dtype): Type to use in computing.
        use_gamma (bool): If ``True``, use scaling parameter. Otherwise, use
            unit(1) which makes no effect.
        use_beta (bool): If ``True``, use shifting parameter. Otherwise, use
            unit(0) which makes no effect.
        communication_backend (str): ``mpi``, ``nccl`` or ``auto``. It is used
            to determine communication backend. If ``auto``, use the best
            communication backend for each communicator.
    """

    def __init__(self, size, comm, decay=0.9, eps=2e-5, dtype=None,
                 use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None,
                 communication_backend='auto'):
        chainer.utils.experimental(
            'chainermn.links.MultiNodeBatchNormalization')

        super(MultiNodeBatchNormalization, self).__init__()
        self._highprec_dtype = chainer.get_dtype(
            dtype, map_mixed16=numpy.float32)
        self.comm = comm
        self.avg_mean = numpy.zeros(size, dtype=self._highprec_dtype)
        self.register_persistent('avg_mean')
        self.avg_var = numpy.zeros(size, dtype=self._highprec_dtype)
        self.register_persistent('avg_var')
        self.N = 0
        self.register_persistent('N')
        self.decay = decay
        self.eps = eps

        self._communication_backend = \
            get_communication_backend(comm, communication_backend)

        with self.init_scope():
            if use_gamma:
                if initial_gamma is None:
                    initial_gamma = 1
                initial_gamma = initializers._get_initializer(initial_gamma)
                initial_gamma.dtype = self._highprec_dtype
                self.gamma = variable.Parameter(initial_gamma, size)
            if use_beta:
                if initial_beta is None:
                    initial_beta = 0
                initial_beta = initializers._get_initializer(initial_beta)
                initial_beta.dtype = self._highprec_dtype
                self.beta = variable.Parameter(initial_beta, size)

    def __call__(self, x, finetune=False):
        if hasattr(self, 'gamma'):
            gamma = self.gamma
        else:
            with cuda.get_device_from_id(self._device_id):
                gamma = variable.Variable(self.xp.ones(
                    self.avg_mean.shape, dtype=self._highprec_dtype))
        if hasattr(self, 'beta'):
            beta = self.beta
        else:
            with cuda.get_device_from_id(self._device_id):
                beta = variable.Variable(self.xp.zeros(
                    self.avg_mean.shape, dtype=self._highprec_dtype))

        if chainer.configuration.config.train:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            func = MultiNodeBatchNormalizationFunction(
                self.comm, self.eps, self.avg_mean, self.avg_var, decay,
                communication_backend=self._communication_backend)

            ret = func(x, gamma, beta)

            self.avg_mean[:] = func.running_mean
            self.avg_var[:] = func.running_var
        else:
            # Use running average statistics or fine-tuned statistics.
            mean = variable.Variable(self.avg_mean)
            var = variable.Variable(self.avg_var)
            ret = batch_normalization.fixed_batch_normalization(
                x, gamma, beta, mean, var, self.eps)
        return ret

    def start_finetuning(self):
        """Resets the population count for collecting population statistics.

        This method can be skipped if it is the first time to use the
        fine-tuning mode. Otherwise, this method should be called before
        starting the fine-tuning mode again.

        """
        self.N = 0
