import numpy

from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check
from chainer.functions.activation import swish

def _sigmoid(x):
    half = x.dtype.type(0.5)
    return numpy.tanh(x * half) * half + half

class ELiSH(function_node.FunctionNode):

    "Exponential Linear Sigmoid SquasHing"

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, inputs):
        x, = inputs
        y = x.copy()
        neg_indices = y<0
        pos_indices = y>=0
        y[pos_indices] = x * _sigmoid(y[pos_indices])
        y[neg_indices] = (numpy.expm1(y[neg_indices])-1) * _sigmoid(y[neg_indices])
        return y,

    def forward_gpu(self, inputs):
        x, = inputs
        y = cuda.elementwise(
                'T x', 'T y', 'y = x>=0 ? x*sigmoid(x) : (expm1(x)-1)*sigmoid(x)',
                'elish_fwd')(x)
        self.retain_outputs((0,))
        return y,

    def backward(self, indexes, grad_outputs):
        raise NotImplementedError()


def elish(x):
    y, = ELiSH().apply((x,))
    return y
