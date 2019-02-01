import chainer
from chainer import configuration
from chainer.functions.normalization import batch_normalization
from chainer.functions.normalization import batch_renormalization
from chainer.links.normalization.batch_normalization import BatchNormalization


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
                 dtype=None, use_gamma=True, use_beta=True,
                 initial_gamma=None, initial_beta=None,
                 initial_avg_mean=None, initial_avg_var=None):
        super(BatchRenormalization, self).__init__(
            size, decay, eps, dtype, use_gamma, use_beta,
            initial_gamma, initial_beta, initial_avg_mean, initial_avg_var)
        self.rmax = rmax  # maximum allowed correction of variance
        self.dmax = dmax  # maximum allowed correction of mean
        self.r = None
        self.d = None

    def forward(self, x, finetune=False):
        if self.gamma is not None:
            gamma = self.gamma
        else:
            with chainer.using_device(self.device):
                gamma = self.xp.ones(
                    self.avg_mean.shape, dtype=x.dtype)

        if self.beta is not None:
            beta = self.beta
        else:
            with chainer.using_device(self.device):
                beta = self.xp.zeros(
                    self.avg_mean.shape, dtype=x.dtype)

        if configuration.config.train:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            avg_mean = self.avg_mean
            avg_var = self.avg_var

            if chainer.config.in_recomputing:
                # Do not update statistics when extra forward computation is
                # called.
                if finetune:
                    self.N -= 1  # Revert the count
                avg_mean = self.xp.zeros_like(self.avg_mean)
                avg_var = self.xp.zeros_like(self.avg_var)

            ret = batch_renormalization.batch_renormalization(
                x, gamma, beta, self.rmax, self.dmax,
                self.eps, avg_mean, avg_var, decay,
                update_statistics=True)
        else:
            # Use running average statistics or fine-tuned statistics.
            mean = self.avg_mean
            var = self.avg_var
            ret = batch_normalization.fixed_batch_normalization(
                x, gamma, beta, mean, var, self.eps)
        return ret
