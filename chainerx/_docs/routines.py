import chainerx
from chainerx import _docs


def set_docs():
    _docs_creation()
    _docs_indexing()
    _docs_linalg()
    _docs_logic()
    _docs_manipulation()
    _docs_math()
    _docs_statistics()
    _docs_connection()
    _docs_normalization()
    _docs_pooling()


def _docs_creation():
    pass


def _docs_indexing():
    pass


def _docs_linalg():
    pass


def _docs_logic():
    _docs.set_doc(
        chainerx.equal,
        """equal(x1, x2)
Returns an elementwise equality array.

Args:
    x1 (~chainerx.ndarray): Input array.
    x2 (~chainerx.ndarray): Input array.

Returns:
    :class:`~chainerx.ndarray`: Output array of type bool.

Note:
    During backpropagation, this function does not propagate gradients.

.. seealso:: :func:`numpy.equal`
""")


def _docs_manipulation():
    _docs.set_doc(
        chainerx.reshape,
        """reshape(a, newshape)
Returns a reshaped array.

Args:
    a (~chainerx.ndarray): Array to be reshaped.
    newshape (int or tuple of ints): The new shape of the array to return.
        If it is an integer, then it is treated as a tuple of length one.
        It should be compatible with ``a.size``. One of the elements can be
        -1, which is automatically replaced with the appropriate value to
        make the shape compatible with ``a.size``.

Returns:
    :class:`~chainerx.ndarray`: A reshaped view of ``a`` if possible,
    otherwise a copy.

Note:
    During backpropagation, this function propagates the gradient of the
    output array to the input array ``a``.

.. seealso:: :func:`numpy.reshape`
""")


def _docs_math():
    pass


def _docs_statistics():
    pass


def _docs_connection():
    pass


def _docs_normalization():
    pass


def _docs_pooling():
    _docs.set_doc(
        chainerx.max_pool,
        """max_pool(x, ksize, stride=None, pad=0, cover_all=False)
Spatial max pooling function.

Args:
    x (~chainerx.ndarray): Input array.
    ksize (int or tuple of ints): Size of pooling window. ``ksize=k`` and
        ``ksize=(k, k, ..., k)`` are equivalent.
    stride (int or tuple of ints or None): Stride of pooling applications.
        ``stride=s`` and ``stride=(s,s, ..., s)`` are equivalent. If
        ``None`` is specified, then it uses same stride as the pooling
        window size.
    pad (int or tuple of ints): Spatial padding width for the input array.
        ``pad=p`` and ``pad=(p, p, ..., p)`` are equivalent.
    cover_all (bool): If ``True``, all spatial locations are pooled into
        some output pixels. It may make the output size larger.

    Returns:
        ~chainerx.ndarray:  Returns the output array.

.. note::

   This function currently does not support ``return_indices`` mode of
   `~chainer.functions.max_pooling_nd`.
""")

    _docs.set_doc(
        chainerx.average_pool,
        """average_pool(x, ksize, stride=None, pad=0, pad_mode="ignore")
Spatial average pooling function.

Args:
    x (~chainerx.ndarray): Input array.
    ksize (int or tuple of ints): Size of pooling window. ``ksize=k`` and
        ``ksize=(k, k, ..., k)`` are equivalent.
    stride (int or tuple of ints or None): Stride of pooling applications.
        ``stride=s`` and ``stride=(s, s, ..., s)`` are equivalent. If
        ``None`` is specified, then it uses same stride as the pooling
        window size.
    pad (int or tuple of ints): Spatial padding width for the input array.
        ``pad=p`` and ``pad=(p, p, ..., p)`` are equivalent.
    pad_mode (str): If ``"zero"`` is specified, the values in the padded
        region are treated as 0.  If ``"ignore"`` is specified, such region
        is ignored.

Returns:
    ~chainerx.ndarray: Output array.
""")
