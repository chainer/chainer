import numpy

from chainer import function
from chainer import link
from chainer.utils import type_check
from chainer import variable


class Parameter(link.Link, function.Function):

    """Function that outputs its weight array.

    This is a parameterized function that takes no input and returns a variable
    holding a shallow copy of the parameter array.

    Args:
        array: Initial parameter array.

    """
    def __init__(self, array):
        super(Parameter, self).__init__()
        self.params['W'] = variable.Variable(array)

    def __call__(self, volatile=False):
        ret = super(Parameter, self).__call__()
        if volatile:
            ret.unchain_backward()
        ret.volatile = volatile
        return ret

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 0)

    def forward(self, x):
        return self.params['W'].data,

    def backward(self, x, gy):
        self.params['W'].grad += gy[0]
        return ()
