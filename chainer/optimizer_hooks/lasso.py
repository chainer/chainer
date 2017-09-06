from chainer import cuda


class Lasso(object):
    """Optimizer/UpdateRule hook function for Lasso regularization.

    This hook function adds a scaled parameter to the sign of each weight.
    It can be used as a regularization.

    Args:
        rate (float): Coefficient for the weight decay.

    Attributes:
        rate (float): Coefficient for the weight decay.

    """
    name = 'Lasso'
    call_for_each_param = True

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, rule, param):
        p, g = param.data, param.grad
        xp = cuda.get_array_module(p)
        with cuda.get_device_from_array(p) as dev:
            sign = xp.sign(p)
            if int(dev) == -1:
                g += self.rate * sign
            else:
                kernel = cuda.elementwise(
                    'T s, T decay', 'T g', 'g += decay * s', 'lasso')
                kernel(sign, self.rate, g)
