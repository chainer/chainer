import numpy

from chainer.utils import argument


def generate_matrix(shape, dtype=float, **kwargs):
    """Generates a random matrix with given properties.

    This function generates a random NumPy matrix (or a set of matrices) that
    has desired properties. It can be used to generate the inputs for a test
    that can be instable when the input value behaves bad.

    Notation: denote the shape of the generated array by :math:`(B..., M, N)`,
    and :math:`K = min\{M, N\}`. :math:`B...` may be an empty sequence.

    Args:
        shape (tuple of int): Shape of the generated array, i.e.,
            :math:`(B..., M, N)`.
        dtype: Dtype of the generated array.
        singular_values (array-like): Singular values of the generated
            matrices. This argument, if given, must be broadcastable to shape
            :math:`(B..., K)`. It cannot be combined with ``condition_number``.
        condition_number (array-like): Upper bound of the condition numbers of
            the generated matrices. The generated matrices have singular values
            in an interval :math:`[1, \\text{condition_number}]`. This
            argument, if given, must be broadcastable to shape :math:`(B...)`.
            It cannot be combined with ``singular_values``.

    """
    singular_values, condition_number = (
        argument.parse_kwargs(
            kwargs,
            singular_values=None,
            condition_number=None,
        )
    )

    if len(shape) <= 1:
        raise ValueError(
            'shpae {} is invalid for matrices: too few axes'.format(shape))
    k_shape = shape[:-2] + (min(shape[-2:]),)
    # TODO(beam2d): consider supporting integer/boolean matrices
    if dtype.kind not in 'fc':
        raise ValueError('dtype {} is not supported'.format(dtype))

    if condition_number is not None:
        if singular_values is not None:
            raise TypeError(
                'condition_number and singular_values cannot be '
                'specified at once'
            )
        if any(condition_number < 1):
            raise ValueError('condition number must not be less than 1')
        singular_values = numpy.random.uniform(
            1, condition_number[..., None], size=k_shape
        )
    if singular_values is None:
        raise TypeError('neither condition_number nor singula_values is given')
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
