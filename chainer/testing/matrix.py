import numpy

from chainer.utils import argument


def generate_matrix(shape, dtype=float, **kwargs):
    r"""generate_matrix(shape, dtype=float, *, singular_values)

    Generates a random matrix with given singular values.

    This function generates a random NumPy matrix (or a stack of matrices) that
    has specified singular values. It can be used to generate the inputs for a
    test that can be instable when the input value behaves bad.

    Notation: denote the shape of the generated array by :math:`(B..., M, N)`,
    and :math:`K = min\{M, N\}`. :math:`B...` may be an empty sequence.

    Args:
        shape (tuple of int): Shape of the generated array, i.e.,
            :math:`(B..., M, N)`.
        dtype: Dtype of the generated array.
        singular_values (array-like): Singular values of the generated
            matrices. It must be broadcastable to shape :math:`(B..., K)`.

    """
    singular_values, = argument.parse_kwargs(
        kwargs, ('singular_values', None),
    )

    if len(shape) <= 1:
        raise ValueError(
            'shape {} is invalid for matrices: too few axes'.format(shape)
        )
    # TODO(beam2d): consider supporting integer/boolean matrices
    dtype = numpy.dtype(dtype)
    if dtype.kind not in 'fc':
        raise TypeError('dtype {} is not supported'.format(dtype))

    if singular_values is None:
        raise TypeError('singular_values is not given')
    singular_values = numpy.asarray(singular_values)
    if not numpy.isrealobj(singular_values):
        raise TypeError('singular_values is not real')
    if (singular_values < 0).any():
        raise ValueError('negative singular value is given')

    if 0 in shape:
        # NumPy<1.16 does not support zero-sized matrices in svd, so skip it.
        # Try broadcast first to raise an error on shape mismatch.
        _broadcast_to(singular_values, shape[:-2] + (min(shape[-2:]),))
        return numpy.empty(shape, dtype=dtype)

    # Generate random matrices with given singular values. We simply generate
    # orthogonal vectors using SVD on random matrices and then combine them
    # with the given singular values.
    a = numpy.random.randn(*shape)
    if dtype.kind == 'c':
        a = a + 1j * numpy.random.randn(*shape)
    u, s, vh = numpy.linalg.svd(a, full_matrices=False)
    sv = _broadcast_to(singular_values, s.shape)
    a = numpy.einsum('...ik,...k,...kj->...ij', u, sv, vh)
    return a.astype(dtype)


def _broadcast_to(array, shape):
    if hasattr(numpy, 'broadcast_to'):
        return numpy.broadcast_to(array, shape)
    # NumPy 1.9 does not support broadcast_to.
    dummy = numpy.empty(shape, dtype=numpy.int8)
    ret, _ = numpy.broadcast_arrays(array, dummy)
    return ret
