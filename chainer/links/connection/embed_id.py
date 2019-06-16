from chainer.functions.connection import embed_id
from chainer.initializers import normal
from chainer import link
from chainer import variable


class EmbedID(link.Link):

    """Efficient linear layer for one-hot input.

    This is a link that wraps the :func:`~chainer.functions.embed_id` function.
    This link holds the ID (word) embedding matrix ``W`` as a parameter.

    Args:
        in_size (int): Number of different identifiers (a.k.a. vocabulary
            size).
        out_size (int): Size of embedding vector.
        initialW (:ref:`initializer <initializer>`): Initializer to
            initialize the weight. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be 2.
        ignore_label (int or None): If ``ignore_label`` is an int value,
            ``i``-th row of return value is filled with ``0``.

    .. seealso:: :func:`~chainer.functions.embed_id`

    Attributes:
        W (~chainer.Variable): Embedding parameter matrix.

    .. admonition:: Example

        >>> W = np.array([[0, 0, 0],
        ...               [1, 1, 1],
        ...               [2, 2, 2]]).astype(np.float32)
        >>> W
        array([[0., 0., 0.],
               [1., 1., 1.],
               [2., 2., 2.]], dtype=float32)
        >>> l = L.EmbedID(W.shape[0], W.shape[1], initialW=W)
        >>> x = np.array([2, 1]).astype(np.int32)
        >>> x
        array([2, 1], dtype=int32)
        >>> y = l(x)
        >>> y.array
        array([[2., 2., 2.],
               [1., 1., 1.]], dtype=float32)

    """

    ignore_label = None

    def __init__(self, in_size, out_size, initialW=None, ignore_label=None):
        super(EmbedID, self).__init__()
        self.ignore_label = ignore_label

        with self.init_scope():
            if initialW is None:
                initialW = normal.Normal(1.0)
            self.W = variable.Parameter(initialW, (in_size, out_size))

    @classmethod
    def from_params(cls, W, ignore_label=None):
        """Initialize `~chainer.links.EmbedID` with the given parameter.

        Args:
            W (:class:`~chainer.Variable` or :ref:`ndarray`):
                The weight parameter.
            ignore_label (int or None): If ``ignore_label`` is an int value,
                ``i``-th column of return value is filled with ``0``.
        """
        in_size, out_size = W.shape
        link = cls(
            in_size, out_size,
            initialW=variable.as_array(W),
            ignore_label=ignore_label
        )
        return link

    def forward(self, x):
        """Extracts the word embedding of given IDs.

        Args:
            x (~chainer.Variable): Batch vectors of IDs.

        Returns:
            ~chainer.Variable: Batch of corresponding embeddings.

        """
        return embed_id.embed_id(x, self.W, ignore_label=self.ignore_label)
