from chainer import utils
from chainer.utils import type_check
from chainer import function_node


def concat_variable(gx, g_input):
    """
    concatenate the inputs to a tuple of variable
    Inputs:
        None
        ~chainer.Variable
        tuple of variable

    Outputs:
        None: When both of gx and g_input is None
        tuple of variable: Otherwise
    """
    sum_gx = ()
    if isinstance(gx, tuple):
        sum_gx = gx
    elif gx is not None:
        sum_gx = gx,

    if isinstance(g_input, tuple):
        sum_gx += g_input
    elif g_input is not None:
        sum_gx += g_input,

    if len(sum_gx) == 0:
        sum_gx = None,

    return sum_gx


class AccumulateAdd(function_node.FunctionNode):

    def check_type_forward(self, in_types):
        for in_type in in_types:
            type_check.expect(
                in_types[0].dtype == in_type.dtype,
                in_types[0].shape == in_type.shape
            )

    def forward(self, xs):
        self.len = len(xs)
        if len(xs) == 1:
            return xs
        # The output should a new array. Add the first 2 arrays
        # and get the result y. Then add the rest arrays to y.
        y = xs[0] + xs[1]
        for x in xs[2:]:
            y += x

        return utils.force_array(y),

    def backward(self, indexes, gy):
        gys = ()
        for i in range(self.len):
            gys += gy[0],
        return gys


def accumulate_add(xs):
    """Element-wise add the input arrays.
    Inputs:
        ~chainer.Variable: Tuple of variable.

    Returns:
        ~chainer.Variable: Output variable.
    """
    return AccumulateAdd().apply(xs)[0]
