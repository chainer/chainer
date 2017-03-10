from chainer import cuda


class GradientHardClipping(object):
    """Optimizer hook function for gradient clipping.

    This hook function clips all gradient arrays to be within a lower and upper
    bound.

    Args:
        lower_bound (float): The lower bound of the gradient value.
        upper_bound (float): The upper bound of the gradient value.

    Attributes:
        lower_bound (float): The lower bound of the gradient value.
        upper_bound (float): The upper bound of the gradient value.

    """
    name = 'GradientHardClipping'

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, opt):
        xp = opt.target.xp
        for param in opt.target.params():
            grad = param.grad
            with cuda.get_device(grad):
                xp.clip(grad, self.lower_bound, self.upper_bound, out=grad)
