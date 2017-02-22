import numpy

from chainer import cuda
from chainer.function.normalization import batch_renormalization
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

    def __init__(self, size, rmax=1, dmax=0, decay=0.9, eps=2e-5,
                 dtype=numpy.float32, use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None, use_cudnn=True):
        super(BatchRenormalization, self).__init__(size, decay, eps, dtype,
                                                   use_gamma, use_beta,
                                                   initial_gamma, initial_beta,
                                                   use_cudnn)
        self.rmax = rmax  # maximum allowed correction of variance
        self.dmax = dmax  # maximum allowed correction of mean

    def __call__(self, x, test=False, finetune=False):
        if hasattr(self, 'gamma'):
            gamma = self.gamma
        else:
            with cuda.get_device(self._device_id):
                gamma = variable.Variable(self.xp.ones(
                    self.avg_mean.shape, dtype=x.dtype), volatile='auto')
        if hasattr(self, 'beta'):
            beta = self.beta
        else:
            with cuda.get_device(self._device_id):
                beta = variable.Variable(self.xp.zeros(
                    self.avg_mean.shape, dtype=x.dtype), volatile='auto')

        if not test:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            func = batch_renormalization.BatchRenormalizationFunction(
                self.eps, self.avg_mean, self.avg_var, True, decay,
                self.use_cudnn, self.rmax, self.dmax)
            ret = func(x, gamma, beta)

            self.avg_mean[:] = func.running_mean
            self.avg_var[:] = func.running_var
        else:
            # Use running average statistics or fine-tuned statistics.
            mean = variable.Variable(self.avg_mean, volatile='auto')
            var = variable.Variable(self.avg_var, volatile='auto')
            ret = batch_renormalization.fixed_batch_normalization(
                x, gamma, beta, mean, var, self.eps, self.use_cudnn)
        return ret
