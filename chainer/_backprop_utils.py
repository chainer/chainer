import chainer


def concat_variable(gx, g_input):
    """concatenate the inputs to a tuple of variable

    Inputs:
        None
        ~chainer.Variable
        tuple of variable

    Outputs:
        None: When both of gx and g_input is None
        Variable: When one is None, and the other is variable
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

    # gx is None and g_input is None
    if len(sum_gx) == 0:
        sum_gx = None
    elif len(sum_gx) == 1:
        sum_gx = sum_gx[0]

    return sum_gx


def add(lhs, rhs):
    y = concat_variable(lhs, rhs)
    return chainer.functions.add(*y)
