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

    def __init__(self, load_if_new=False):
        self.grads = {}
        self._load_if_new = load_if_new

    def __setitem__(self, node, grad):
        self.grads[node] = _pure(grad)

    def get_as_list(self, node):
        if node is None:
            return []
        grads = self.grads
        if node not in grads:
            if self._load_if_new and node.creator_node is None:
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
        if self._load_if_new:
            return node.grad_var
        else:
            return []
