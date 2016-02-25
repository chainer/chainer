import chainer
from chainer import function


class PrintHook(function.FunctionHook):

    name = 'PrintHook'

    def __call__(self, function, in_data, out_grad=None):
        print('function\t{}'.format(function.label))
        print('input data')
        for d in in_data:
            print(chainer.Variable(d).debug_print())
        if out_grad is not None:
            print('output gradient')
            for d in out_grad:
                v = chainer.Variable.empty_like(d, dtype=d.dtype)
                v.grad = d
                print(v.debug_print())
