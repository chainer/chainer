from chainer import cuda


class Lasso(object):
    """Optimizer hook function for Lasso regularization.

    This hook function adds a scaled parameter to the sign of each weight.
    It can be used as a regularization.

    Args:
        rate (float): Coefficient for the weight decay.

    Attributes:
        rate (float): Coefficient for the weight decay.

    """
    name = 'Lasso'

    def __init__(self, rate):
        self.rate = rate

    def kernel(self):
        return cuda.elementwise(
            'T s, T decay', 'T g', 'g += decay * s', 'lasso')

    def __call__(self, opt):
        rate = self.rate
        for param in opt.target.params():
            p, g = param.data, param.grad
            xp = cuda.get_array_module(p)
            sign = xp.sign(p)
            with cuda.get_device(p) as dev:
                if int(dev) == -1:
                    g += rate * sign
                else:
                    self.kernel()(sign, rate, g)
