from chainer import backend
from chainer import cuda


class GradientHardClipping(object):

    """Optimizer/UpdateRule hook function for gradient clipping.

    This hook function clips all gradient arrays to be within a lower and upper
    bound.

    Args:
        lower_bound (float): The lower bound of the gradient value.
        upper_bound (float): The upper bound of the gradient value.

    Attributes:
        ~optimizer_hooks.GradientHardClipping.lower_bound (float): The
                         lower bound of the gradient value.
        ~optimizer_hooks.GradientHardClipping.upper_bound (float): The
                         upper bound of the gradient value.
        ~optimizer_hooks.GradientHardClipping.timing (string): Specifies
                         when this hook should be called by the
                         Optimizer/UpdateRule. Valid values are 'pre'
                         (before any updates) and 'post'
                         (after any updates).
        ~optimizer_hooks.GradientHardClipping.call_for_each_param (bool): \
                         Specifies if this hook is called for each parameter
                         (``True``) or only once (``False``) by an optimizer to
                         which this hook is registered. This function does
                         not expect users to switch the value from default one,
                         which is `True`.

    .. versionadded:: 4.0.0
       The *timing* parameter.

    """
    name = 'GradientHardClipping'
    call_for_each_param = True
    timing = 'pre'

    def __init__(self, lower_bound, upper_bound):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, rule, param):
        grad = param.grad
        if grad is None:
            return
        xp = backend.get_array_module(grad)
        with cuda.get_device_from_array(grad):
            xp.clip(grad, self.lower_bound, self.upper_bound, out=grad)
