import collections

import numpy
import six

from chainer import cuda


def _sum_sqnorm(arr):
    sq_sum = collections.defaultdict(float)
    for x in arr:
        with cuda.get_device_from_array(x) as dev:
            x = x.ravel()
            s = x.dot(x)
            sq_sum[int(dev)] += s
    return sum([float(i) for i in six.itervalues(sq_sum)])


class GradientClipping(object):
    """Optimizer hook function for gradient clipping.

    This hook function scales all gradient arrays to fit to the defined L2 norm
    threshold.

    Args:
        threshold (float): L2 norm threshold.

    Attributes:
        ~optimizer_hooks.GradientClipping.threshold (float): L2
                         norm threshold of gradient norm.
        ~optimizer_hooks.GradientClipping.timing (string): Specifies
                         when this hook should be
                         called by the Optimizer/UpdateRule. Valid values are
                         'pre' (before any updates) and 'post' (after any
                         updates).

    .. versionadded:: 4.0.0
       The *timing* parameter.

    """
    name = 'GradientClipping'
    timing = 'pre'

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, opt):
        norm = numpy.sqrt(_sum_sqnorm(
            [p.grad for p in opt.target.params(False)]))
        rate = self.threshold / norm
        if rate < 1:
            for param in opt.target.params(False):
                grad = param.grad
                with cuda.get_device_from_array(grad):
                    grad *= rate
