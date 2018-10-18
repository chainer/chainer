import numpy

from chainer import backend
from chainer import initializer


# Original code forked from MIT licensed keras project
# https://github.com/fchollet/keras/blob/master/keras/initializations.py

class Orthogonal(initializer.Initializer):
    """Initializes array with an orthogonal system.

    This initializer first makes a matrix of the same shape as the
    array to be initialized whose elements are drawn independently from
    standard Gaussian distribution.
    Next, it applies Singular Value Decomposition (SVD) to the matrix.
    Then, it initializes the array with either side of resultant
    orthogonal matrices, depending on the shape of the input array.
    Finally, the array is multiplied by the constant ``scale``.

    If the ``ndim`` of the input array is more than 2, we consider the array
    to be a matrix by concatenating all axes except the first one.

    The number of vectors consisting of the orthogonal system
    (i.e. first element of the shape of the array) must be equal to or smaller
    than the dimension of each vector (i.e. second element of the shape of
    the array).

    Attributes:
        scale (float): A constant to be multiplied by.
        dtype: Data type specifier.

    Reference: Saxe et al., https://arxiv.org/abs/1312.6120

    """

    def __init__(self, scale=1.1, dtype=None):
        self.scale = scale
        super(Orthogonal, self).__init__(dtype)

    # TODO(Kenta Oono)
    # How do we treat overcomplete base-system case?
    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype
        xp = backend.get_array_module(array)
        if not array.shape:  # 0-dim case
            array[...] = self.scale * (2 * numpy.random.randint(2) - 1)
        elif not array.size:
            raise ValueError('Array to be initialized must be non-empty.')
        else:
            # numpy.prod returns float value when the argument is empty.
            flat_shape = (len(array), int(numpy.prod(array.shape[1:])))
            if flat_shape[0] > flat_shape[1]:
                raise ValueError('Cannot make orthogonal system because'
                                 ' # of vectors ({}) is larger than'
                                 ' that of dimensions ({})'.format(
                                     flat_shape[0], flat_shape[1]))
            a = numpy.random.normal(size=flat_shape)
            # we do not have cupy.linalg.svd for now
            u, _, vh = numpy.linalg.svd(a, full_matrices=False)
            # pick the one with the correct shape
            if u.shape == flat_shape:
                q = u * numpy.sign(numpy.diag(vh))[None, :]
            else:
                q = vh * numpy.sign(numpy.diag(u))[:, None]
            array[...] = xp.asarray(q.reshape(array.shape))
            array *= self.scale
