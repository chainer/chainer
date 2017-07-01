from chainer import cuda


class GradientHardClipping(object):

    """Optimizer/UpdateRule hook function for gradient clipping.

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
    call_for_each_param = True

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, rule, param):
        grad = param.grad
        xp = cuda.get_array_module(grad)
        with cuda.get_device_from_array(grad):
            xp.clip(grad, self.lower_bound, self.upper_bound, out=grad)
