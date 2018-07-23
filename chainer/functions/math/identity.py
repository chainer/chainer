from chainer import function_node


class Identity(function_node.FunctionNode):

    """Identity function."""

    def forward(self, xs):
        return xs

    def backward(self, indexes, gys):
        return gys


def identity(*inputs):
    """Just returns input variables."""
    ret = Identity().apply(inputs)
    return ret[0] if len(ret) == 1 else ret
