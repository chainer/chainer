import numpy

from chainer.utils import argument


def generate_matrix(shape, dtype=float, **kwargs):
    """Generates a random matrix with given singular values.

    This function generates a random NumPy matrix (or a set of matrices) that
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
        kwargs, singular_values=None,
    )

    if len(shape) <= 1:
        raise ValueError(
            'shpae {} is invalid for matrices: too few axes'.format(shape))
    k_shape = shape[:-2] + (min(shape[-2:]),)
    # TODO(beam2d): consider supporting integer/boolean matrices
    if dtype.kind not in 'fc':
        raise ValueError('dtype {} is not supported'.format(dtype))

    if singular_values is None:
        raise TypeError('singular_values is not given')
    singular_values = numpy.asarray(singular_values)
    if any(singular_values < 0):
        raise ValueError('negative singular value is given')

    # Generate random matrices with given singular values. We simply generate
    # orthogonal vectors using SVD on random matrices and then combine them
    # with the given singular values.
    a = numpy.random.randn(*shape)
    if dtype.kind == 'c':
        a += 1j * numpy.random.randn(*shape)
    u, _, vh = numpy.linalg.svd(a, full_matrices=False)
    a = numpy.einsum('...ik,...k,...kj->...ij', u, singular_values, vh)
    return a.astype(dtype, copy=False)
