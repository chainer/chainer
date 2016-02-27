from chainer import function


class CallableHook(function.FunctionHook):
    """Wrapper function hook of the callable object."""

    name = 'CallableHook'

    def __init__(self, f):
        """
        Args:
            f(callable): A callable object for preprocess
                and postprocess. The ``f`` must be able to call
                with arguments ``function``, ``in_data``, and ``out_grad``.
        """
        self.f = f

    def __call__(self, function, in_data, out_grad=None):
        self.f(function, in_data, out_grad)
