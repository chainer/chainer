from chainer.functions.loss import crf1d
from chainer import link
from chainer import variable


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
        super(CRF1d, self).__init__()
        with self.init_scope():
            self.cost = variable.Parameter(0, (n_label, n_label))

    def __call__(self, xs, ys, reduce='mean'):
        return crf1d.crf1d(self.cost, xs, ys, reduce)

    def argmax(self, xs):
        """Computes a state that maximizes a joint probability.

        Args:
            xs (list of Variable): Input vector for each label.

        Returns:
            tuple: A tuple of :class:`~chainer.Variable` representing each
                log-likelihood and a list representing the argmax path.

        .. seealso:: See :func:`~chainer.frunctions.crf1d_argmax` for more
           detail.

        """
        return crf1d.argmax_crf1d(self.cost, xs)
