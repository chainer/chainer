from chainer.functions.loss import crf1d
from chainer import link


class CRF1d(link.Link):

    """Linear-chain conditional random field loss layer.

    This link wraps the :func:`~chainer.functions.crf1d` function.
    It holds a transition cost matrix as a parameter.

    Args:
        n_label (int): Number of labels.

    .. seealso:: :func:`~chainer.functions.crf1d` for more detail.

    Attributes:
        cost (~chainer.Variable): Transition cost parameter.
    """

    def __init__(self, n_label):
        super(CRF1d, self).__init__(cost=(n_label, n_label))
        self.cost.data[...] = 0

    def __call__(self, xs, ys):
        return crf1d.crf1d(self.cost, xs, ys)

    def argmax(self, xs):
        return crf1d.argmax_crf1d(self.cost, xs)
