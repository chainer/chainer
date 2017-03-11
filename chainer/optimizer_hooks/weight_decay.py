from chainer import cuda


class WeightDecay(object):
    """Optimizer hook function for weight decay regularization.

    This hook function adds a scaled parameter to the corresponding gradient.
    It can be used as a regularization.

    Args:
        rate (float): Coefficient for the weight decay.

    Attributes:
        rate (float): Coefficient for the weight decay.

    """
    name = 'WeightDecay'

    def __init__(self, rate):
        self.rate = rate

    def kernel(self):
        return cuda.elementwise(
            'T p, T decay', 'T g', 'g += decay * p', 'weight_decay')

    def __call__(self, opt):
        rate = self.rate
        for param in opt.target.params():
            p, g = param.data, param.grad
            with cuda.get_device(p) as dev:
                if int(dev) == -1:
                    g += rate * p
                else:
                    self.kernel()(p, rate, g)
