import sys

import chainer
from chainer import function


class PrintHook(function.FunctionHook):

    name = 'PrintHook'

    def __call__(self, function, in_data, out_grad=None, file=sys.stdout):
        print('function\t{}'.format(function.label), file=file)
        print('input data', file=file)
        for d in in_data:
            print(chainer.Variable(d).debug_print(), file=file)
        if out_grad is not None:
            print('output gradient', file=file)
            for d in out_grad:
                v = chainer.Variable.empty_like(d, dtype=d.dtype)
                v.grad = d
                print(v.debug_print(), file=file)
