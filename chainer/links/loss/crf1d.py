from chainer.functions.loss import crf1d
from chainer.functions.array import transpose_sequence
from chainer.links.connection.n_step_rnn import argsort_list_descent
from chainer.links.connection.n_step_rnn import permutate_list
from chainer import initializers
from chainer import link
from chainer import variable


class CRF1d(link.Link):

    """Linear-chain conditional random field loss layer.

    This link wraps the :func:`~chainer.functions.crf1d` function.
    It holds a transition cost matrix as a parameter.

    Args:
        n_label (int): Number of labels.
        initial_cost (:ref:`initializer <initializer>`): Initializer to
            initialize the transition cost matrix.
            If this attribute is not specified,
            the transition cost matrix is initialized with zeros.

    .. seealso:: :func:`~chainer.functions.crf1d` for more detail.

    Attributes:
        cost (~chainer.Variable): Transition cost parameter.
    """

    def __init__(self, n_label, initial_cost=None):
        super(CRF1d, self).__init__()
        if initial_cost is None:
            initial_cost = initializers.constant.Zero()

        with self.init_scope():
            self.cost = variable.Parameter(initializer=initial_cost,
                                           shape=(n_label, n_label))

    def forward(self, xs, ys, reduce='mean', transpose=False):
        """Computes negative log-likelihood of linear-chain CRF

        Args:
            xs (list of Variable): Input vector for each label
            ys (list of Variable): Expected output labels.
            transpose (bool): If ``True``, input/output sequences
            will be sorted in descending order of length.

        Returns:
            ~chainer.Variable: A variable holding the average negative
            log-likelihood of the input sequences.

        .. seealso:: See :func:`~chainer.frunctions.crf1d` for more detail.
        """

        if transpose:
            indices = argsort_list_descent(xs)
            xs = permutate_list(xs, indices, inv=False)
            ys = permutate_list(ys, indices, inv=False)
            trans_x = transpose_sequence.transpose_sequence(xs)
            trans_y = transpose_sequence.transpose_sequence(ys)
            loss = crf1d.crf1d(self.cost, trans_x, trans_y, reduce)

        else:
            loss = crf1d.crf1d(self.cost, xs, ys, reduce)

        return loss

    def argmax(self, xs, transpose=False):
        """Computes a state that maximizes a joint probability.

        Args:
            xs (list of Variable): Input vector for each label.
            transpose (bool): If ``True``, input/output sequences
            will be sorted in descending order of length.

        Returns:
            tuple: A tuple of :class:`~chainer.Variable` representing each
            log-likelihood and a list representing the argmax path.

        .. seealso:: See :func:`~chainer.frunctions.crf1d_argmax` for more
           detail.

        """

        if transpose:
            indices = argsort_list_descent(xs)
            xs = permutate_list(xs, indices, inv=False)
            trans_x = transpose_sequence.transpose_sequence(xs)
            score, path = crf1d.argmax_crf1d(self.cost, trans_x)

            path = transpose_sequence.transpose_sequence(path)
            path = [p.array for p in path]
            path = permutate_list(path, indices, inv=True)

        else:
            score, path = crf1d.argmax_crf1d(self.cost, xs)

        return score, path
