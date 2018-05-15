import chainer


def normalize(grad_list):
    if not grad_list:
        return None
    if len(grad_list) >= 2:
        grad_list[:] = [chainer.functions.add(*grad_list)]
    return grad_list[0]


def _pure(grad):
    return [] if grad is None else [grad]


class GradTable(object):

    def __init__(self):
        self.grads = {}

    def __setitem__(self, node, grad):
        self.grads[node] = _pure(grad)

    def get_as_list(self, node):
        if node is None:
            return []
        grads = self.grads
        if node not in grads:
            if node.creator_node is None:
                node._check_old_style_gradient()
                # accumulate the gradient only if the node is a leaf
                grads[node] = _pure(node.grad_var)
            else:
                grads[node] = []
        return grads[node]

    def pop(self, node):
        if node is None:
            return None
        grads = self.grads
        if node in grads:
            return normalize(grads.pop(node))
        return node.grad_var


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
