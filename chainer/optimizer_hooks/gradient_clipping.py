import numpy

from chainer import cuda


class GradientClipping(object):
    """Optimizer hook function for gradient clipping.

    This hook function scales all gradient arrays to fit to the defined L2 norm
    threshold.

    Args:
        threshold (float): L2 norm threshold.

    Attributes:
        threshold (float): L2 norm threshold of gradient norm.

    """
    name = 'GradientClipping'

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, opt):
        norm = numpy.sqrt(_sum_sqnorm([p.grad for p in opt.target.params()]))
        rate = self.threshold / norm
        if rate < 1:
            for param in opt.target.params():
                grad = param.grad
                with cuda.get_device(grad):
                    grad *= rate
