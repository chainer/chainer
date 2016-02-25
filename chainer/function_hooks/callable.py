from chainer import function


class CallableHook(function.FunctionHook):

    name = 'CallableHook'

    def __init__(self, f, name=None):
        self.f = f
        if name is not None:
            self.name = name

    def __call__(self, function, in_data, out_grad=None):
        self.f(function, in_data, out_grad)
