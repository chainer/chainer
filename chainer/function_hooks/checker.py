from chainer import cuda
from chainer import function


class CheckerHook(function.FunctionHook):

    def __init__(self, checker):
        self.checker = checker

    def __call__(self, function, in_data, out_grad=None):
        if self.checker(in_data):
            print('Detected in data')

        if out_grad is not None and self.checker(out_grad):
            print('Detected out grad')


def isnan(data):
    xp = cuda.get_array_module(*data)
    flags = [xp.isnan(d).any() for d in data]
    return any(flags)


def nan_checker():
    return CheckerHook(isnan)


def is_equal(data, id_):
    flags = [d.dtype.kind != 'i' and (d == id_).any() for d in data]
    return any(flags)


def id_checker(id_):
    return CheckerHook(lambda data: is_equal(data, id_))
