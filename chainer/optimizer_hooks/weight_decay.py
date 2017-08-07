from chainer import cuda


class WeightDecay(object):

    """Optimizer/UpdateRule hook function for weight decay regularization.

    This hook function adds a scaled parameter to the corresponding gradient.
    It can be used as a regularization.

    Args:
        rate (float): Coefficient for the weight decay.

    Attributes:
        rate (float): Coefficient for the weight decay.

    """
    name = 'WeightDecay'
    call_for_each_param = True

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, rule, param):
        p, g = param.data, param.grad
        with cuda.get_device_from_array(p) as dev:
            if int(dev) == -1:
                g += self.rate * p
            else:
                kernel = cuda.elementwise(
                    'T p, T decay', 'T g', 'g += decay * p', 'weight_decay')
                kernel(p, self.rate, g)
