import sys

import chainer
from chainer import function


class PrintHook(function.FunctionHook):
    """Function hook that prints debug information.

    Attributes are same as the keyword argument of print function.

    Attributes:
        sep: Separator of print function.
        end: Character to be added at the end of print function.
        file: Output file_like object that that redirect to.
        flush: If ``True``, print function forcibly flushes the text stream.
    """

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
