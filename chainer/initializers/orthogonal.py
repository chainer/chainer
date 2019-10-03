import numpy

from chainer import backend
from chainer import initializer
from chainer import utils
from chainer.utils import argument


_orthogonal_constraints = {  # (assert emb., assert proj.)
    'auto': (False, False),
    'projection': (False, True),
    'embedding': (True, False),
    'basis': (True, True),
}


# Original code forked from MIT licensed keras project
# https://github.com/fchollet/keras/blob/master/keras/initializations.py

class Orthogonal(initializer.Initializer):
    """Initializes array with an orthogonal system.

    This initializer first makes a matrix of the same shape as the
    array to be initialized whose elements are drawn independently from
    standard Gaussian distribution.
    Next, it applies QR decomposition to (the transpose of) the matrix.
    To make the decomposition (almost surely) unique, we require the diagonal
    of the triangular matrix R to be non-negative (see e.g. Edelman & Rao,
    https://web.eecs.umich.edu/~rajnrao/Acta05rmt.pdf).
    Then, it initializes the array with the (semi-)orthogonal matrix Q.
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
        mode (str): Assertion on the initialized shape.
            ``'auto'`` (default), ``'projection'`` (before v7),
            ``'embedding'``, or ``'basis'``.
        rng (xp.random.RandomState): Pseudo-random number generator.

    Reference: Saxe et al., https://arxiv.org/abs/1312.6120

    """

    def __init__(self, scale=1.1, dtype=None, mode='auto', **kwargs):
        self.scale = scale
        self.mode = mode
        rng = None
        if kwargs:
            rng, = argument.parse_kwargs(kwargs, ('rng', rng))
        self.rng = rng
        try:
            self._checks = _orthogonal_constraints[mode]
        except KeyError:
            raise ValueError(
                'Invalid mode: {}. Choose from {}.'.format(
                    repr(mode),
                    ', '.join(repr(m) for m in _orthogonal_constraints)))
        super(Orthogonal, self).__init__(dtype)

    # TODO(Kenta Oono)
    # How do we treat overcomplete base-system case?
    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype,\
                '{} != {}'.format(array.dtype, self.dtype)
        if not array.shape:  # 0-dim case
            if self.rng is None:
                a = numpy.random.randint(2)
            else:
                a = self.rng.randint(2)
            a = int(a)
            array[...] = self.scale * (2 * a - 1)
        elif not array.size:
            raise ValueError('Array to be initialized must be non-empty.')
        else:
            # numpy.prod returns float value when the argument is empty.
            out_dim = len(array)
            in_dim = utils.size_of_shape(array.shape[1:])
            if (in_dim > out_dim and self._checks[0]) or (
                    in_dim < out_dim and self._checks[1]):
                raise ValueError(
                    'Cannot make orthogonal {}. '
                    'shape = {}, interpreted as '
                    '{}-dim input and {}-dim output.'.format(
                        self.mode, array.shape, in_dim, out_dim))
            transpose = in_dim > out_dim
            if self.rng is None:
                a = numpy.random.normal(size=(out_dim, in_dim))
            else:
                a_tmp = self.rng.normal(size=(out_dim, in_dim))
                a = numpy.empty(a_tmp.shape, dtype=a_tmp.dtype)
                backend.copyto(a, a_tmp)
            if transpose:
                a = a.T
            # cupy.linalg.qr requires cusolver in CUDA 8+
            q, r = numpy.linalg.qr(a)
            q *= numpy.copysign(self.scale, numpy.diag(r))
            if transpose:
                q = q.T
            backend.copyto(array, q.reshape(array.shape).astype(
                array.dtype, copy=False))
