import numpy

from chainer import configuration
from chainer import functions
from chainer import link
from chainer.utils import argument
from chainer import variable


class DecorrelatedBatchNormalization(link.Link):

    def __init__(self, size, groups=16, decay=0.9, eps=2e-5,
                 dtype=numpy.float32):
        super(DecorrelatedBatchNormalization, self).__init__()
        self.expected_mean = numpy.zeros(size // groups, dtype=dtype)
        self.register_persistent('expected_mean')
        self.expected_projection = numpy.eye(size // groups, dtype=dtype)
        self.register_persistent('expected_projection')
        self.N = 0
        self.register_persistent('N')
        self.decay = decay
        self.eps = eps
        self.groups = groups

    def __call__(self, x, **kwargs):
        finetune, = argument.parse_kwargs(kwargs, ('finetune', False))

        if configuration.config.train:
            if finetune:
                self.N += 1
                decay = 1. - 1. / self.N
            else:
                decay = self.decay

            ret = functions.decorrelated_batch_normalization(
                x, groups=self.groups, eps=self.eps,
                expected_mean=self.expected_mean,
                expected_projection=self.expected_projection, decay=decay)
        else:
            # Use running average statistics or fine-tuned statistics.
            mean = variable.Variable(self.expected_mean)
            projection = variable.Variable(self.expected_projection)
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
