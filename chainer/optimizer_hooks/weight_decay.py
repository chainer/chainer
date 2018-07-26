from chainer import cuda


class WeightDecay(object):

    """Optimizer/UpdateRule hook function for weight decay regularization.

    This hook function adds a scaled parameter to the corresponding gradient.
    It can be used as a regularization.

    Args:
        rate (float): Coefficient for the weight decay.

    Attributes:
        ~optimizer_hooks.WeightDecay.rate (float): Coefficient
                         for the weight decay.
        ~optimizer_hooks.WeightDecay.timing (string): Specifies
                         when this hook should be called by the
                         Optimizer/UpdateRule. Valid values are 'pre'
                         (before any updates) and 'post' (after any updates).
        ~optimizer_hooks.WeightDecay.call_for_each_param (bool): Specifies
                         if this hook is called for each parameter (``True``)
                         or only once (``False``) by an optimizer to
                         which this hook is registered. This function does
                         not expect users to switch the value from default one,
                         which is `True`.

    .. versionadded:: 4.0.0
       The *timing* parameter.

    """
    name = 'WeightDecay'
    call_for_each_param = True
    timing = 'pre'

    def __init__(self, rate):
        self.rate = rate

    def __call__(self, rule, param):
        p, g = param.data, param.grad
        if p is None or g is None:
            return
        with cuda.get_device_from_array(p) as dev:
            if int(dev) == -1:
                g += self.rate * p
            else:
                kernel = cuda.elementwise(
                    'T p, T decay', 'T g', 'g += decay * p', 'weight_decay')
                kernel(p, self.rate, g)
