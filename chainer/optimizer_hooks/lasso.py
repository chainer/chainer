import chainer
from chainer import cuda


class Lasso(object):
    """Optimizer/UpdateRule hook function for Lasso regularization.

    This hook function adds a scaled parameter to the sign of each weight.
    It can be used as a regularization.

    Args:
        rate (float): Coefficient for the weight decay.

    Attributes:
        ~optimizer_hooks.Lasso.rate (float): Coefficient for the weight decay.
        ~optimizer_hooks.Lasso.timing (string): Specifies
                         when this hook should be called by
                         the Optimizer/UpdateRule. Valid values are 'pre'
                         (before any updates) and 'post' (after any updates).
        ~optimizer_hooks.Lasso.call_for_each_param (bool): Specifies
                         if this hook is called for each parameter (``True``)
                         or only once (``False``) by an optimizer to
                         which this hook is registered. This function does
                         not expect users to switch the value from default one,
                         which is `True`.

    .. versionadded:: 4.0.0
       The *timing* parameter.

    """
    name = 'Lasso'
    call_for_each_param = True
    timing = 'pre'

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, rule, param):
        p, g = param.data, param.grad
        if p is None or g is None:
            return
        with chainer.using_device(param.device):
            xp = param.device.xp
            sign = xp.sign(p)
            if xp is cuda.cupy:
                kernel = cuda.elementwise(
                    'T s, T decay', 'T g', 'g += decay * s', 'lasso')
                kernel(sign, self.rate, g)
            else:
                g += self.rate * sign
