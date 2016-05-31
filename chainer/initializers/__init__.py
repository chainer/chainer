import numpy

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
