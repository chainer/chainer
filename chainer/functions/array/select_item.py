import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class SelectItem(function.Function):

    """Select elements stored in given indicies."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, t_type = in_types
        type_check.expect(
            x_type.dtype.kind == 'f',
            t_type.dtype.kind == 'i',
            x_type.ndim == 2,
            t_type.ndim == 1,
            x_type.shape[0] == t_type.shape[0],
        )

    def forward_cpu(self, inputs):
        x, t = inputs
        return t.choose(x.T),

    def forward_gpu(self, inputs):
        x, t = inputs
        y = cuda.elementwise(
            'S t, raw T x',
            'T y',
            'int ind[] = {i, t}; y = x[ind];',
            'getitem_fwd'
        )(t, x)
        return y,

    def backward_cpu(self, inputs, grad_outputs):
        x, t = inputs
        gloss = grad_outputs[0]
        gx = numpy.zeros_like(x)
        gx[six.moves.range(t.size), t] = gloss
        return gx, None

    def backward_gpu(self, inputs, grad_outputs):
        x, t = inputs
        gloss = grad_outputs[0]
        gx = cuda.cupy.zeros_like(x)
        gx = cuda.elementwise(
            'S t, T gloss',
            'raw T gx',
            'int ind[] = {i, t}; gx[ind] = gloss;',
            'getitem_bwd'
        )(t, gloss, gx)
        return gx, None


def select_item(x, t):
    """Select elements stored in given indicies.

    This function returns ```t.choose(x.T)```, that means
    ```y[i] == x[i, t[i]]``` for all ```i```.

    Args:
        x (Variable): Variable storing arrays.
        t (Variable): Variable storing index numbers.

    Returns:
        ~chainer.Variable: Variable that holds ```t```-th element of ```x```.

    """
    return SelectItem()(x, t)
