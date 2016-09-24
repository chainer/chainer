import numpy

from chainer import initializer
from chainer.initializers import constant
from chainer.initializers import normal
from chainer.initializers import orthogonal
from chainer.initializers import uniform


Identity = constant.Identity
Constant = constant.Constant
Zero = constant.Zero
One = constant.One
Normal = normal.Normal
GlorotNormal = normal.GlorotNormal
HeNormal = normal.HeNormal
Orthogonal = orthogonal.Orthogonal
Uniform = uniform.Uniform
LeCunUniform = uniform.LeCunUniform
GlorotUniform = uniform.GlorotUniform
HeUniform = uniform.HeUniform


def generate_array(initializer, shape, xp):
    """Return initialized array.

    The algorithms used to make the new values depend on the
    concrete derived classes. The dtype of a generated array depends on
    ``initializer.dtype``.

    Args:
        initializer: A callable object that takes :class:`numpy.ndarray`
             or :class:`cupy.ndarray` and edits its value.
        shape (tuple): Shape of a return array.
        xp (module): :mod:`cupy` or :mod:`numpy`.

    Returns:
        numpy.ndarray or cupy.ndarray: An initialized array.

    """
    dtype = numpy.float32
    if hasattr(initializer, 'dtype') and initializer.dtype is not None:
        dtype = initializer.dtype
    array = xp.empty(shape, dtype=dtype)
    initializer(array)
    return array


def init_weight(weights, initializer, scale=1.0):
    """Helper function for initialization of the weight tensor.

    This function accepts several types of initializer, prepares
    the appropriate ``~chainer.Initializer`` if necessary,
    and does the initialization.

    Args:
         weights (numpy.ndarray or cupy.ndarray):
             Weight tensor to be initialized.
         initializer: The value used to initialize the data.
             May be ``None`` (in which case
             :class:`~chainer.initializers.HeNormal`
             is used as an initializer), a scalar to set all values to,
             an ``numpy.ndarray`` to be assigned,
             or a callable that takes :class:`numpy.ndarray`
             or :class:`cupy.ndarray` and edits its value.
         scale (scalar): A constant to multiply initializer by.

    """

    if initializer is None:
        initializer = HeNormal(1 / numpy.sqrt(2))
    elif numpy.isscalar(initializer):
        initializer = Constant(initializer)
    elif isinstance(initializer, numpy.ndarray):
        initializer = Constant(initializer)

    assert callable(initializer)
    initializer(weights)
    weights *= scale


class _ScaledInitializer(initializer.Initializer):

    def __init__(self, initializer, scale=1.0):
        self.initializer = initializer
        self.scale = scale
        dtype = getattr(initializer, 'dtype', None)
        super(Identity, self).__init__(dtype)

    def __call__(self, array):
        self.initializer(array)
        array *= self.scale


def _get_initializer(initializer, scale=1.0):
    if initializer is None:
        return HeNormal(scale / numpy.sqrt(2))
    if numpy.isscalar(initializer):
        return Constant(initializer * scale)
    if isinstance(initializer, numpy.ndarray):
        return Constant(initializer * scale)

    assert callable(initializer)
    if scale == 1.0:
        return initializer
    return _ScaledInitializer(initializer, scale)
