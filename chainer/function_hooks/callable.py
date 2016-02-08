


class CallableHook(FunctionHook):

    def __init__(self, f):
        self.f = f

    def __call__(self, function):
        return self.f(function)
