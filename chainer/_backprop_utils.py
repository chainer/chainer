def concat_variable(gx, g_input):
    """concatenate the inputs to a tuple of variable

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
