import numpy

from chainer.backends import cuda
from chainer import configuration
from chainer.functions.normalization import batch_renormalization
from chainer.links.normalization.batch_normalization import BatchNormalization
from chainer import variable


class BatchRenormalization(BatchNormalization):

    """Batch renormalization layer on outputs of linear or convolution functions.

    This link wraps the :func:`~chainer.functions.batch_renormalization` and
    :func:`~chainer.functions.fixed_batch_renormalization` functions.

    This is an extension of batch normalization, which ensures that the
    training and inference models generate the same outputs that depend on
    individual examples rather than the entire minibatch.

    See: `Batch Renormalization: Towards Reducing Minibatch Dependence in \
          Batch-Normalized Models <https://arxiv.org/abs/1702.03275>`_

    .. seealso::
       :func:`~chainer.functions.batch_renormalization`,
       :func:`~chainer.functions.fixed_batch_renormalization`
       :func:`~chainer.functions.batch_normalization`,

    """

    gamma = None
    beta = None

    def __init__(self, size, rmax=1, dmax=0, decay=0.9, eps=2e-5,
                 dtype=numpy.float32, use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None,
                 freeze_running_statistics=False):
        super(BatchRenormalization, self).__init__(size, decay, eps, dtype,
                                                   use_gamma, use_beta,
                                                   initial_gamma, initial_beta)
        self.rmax = rmax  # maximum allowed correction of variance
        self.dmax = dmax  # maximum allowed correction of mean
        self.r = None
        self.d = None
        self.freeze_running_statistics = freeze_running_statistics

    def __call__(self, x, finetune=False):
        if self.gamma is not None:
            gamma = self.gamma
        else:
            with cuda.get_device_from_id(self._device_id):
                gamma = variable.Variable(self.xp.ones(
                    self.avg_mean.shape, dtype=x.dtype))

        if self.beta is not None:
            beta = self.beta
        else:
            with cuda.get_device_from_id(self._device_id):
                beta = variable.Variable(self.xp.zeros(
                    self.avg_mean.shape, dtype=x.dtype))

        if configuration.config.train:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            func = batch_renormalization.BatchRenormalizationFunction(
                self.eps, self.avg_mean, self.avg_var, decay,
                self.rmax, self.dmax, self.freeze_running_statistics)
            if self.freeze_running_statistics:
                func.r = self.r
                func.d = self.d
            ret = func(x, gamma, beta)
            if self.freeze_running_statistics and self.r is None:
                self.r = func.r
                self.d = func.d

            self.avg_mean[:] = func.running_mean
            self.avg_var[:] = func.running_var
        else:
            # Use running average statistics or fine-tuned statistics.
            mean = variable.Variable(self.avg_mean)
            var = variable.Variable(self.avg_var)
            ret = batch_renormalization.fixed_batch_renormalization(
                x, gamma, beta, mean, var, self.eps)
        return ret
