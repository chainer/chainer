import numpy

from chainer import backend
from chainer import initializer
from chainer.utils import argument


# Original code forked from MIT licensed keras project
# https://github.com/fchollet/keras/blob/master/keras/initializations.py

class Uniform(initializer.Initializer):

    """Initializes array with a scaled uniform distribution.

    Each element of the array is initialized by the value drawn
    independently from uniform distribution :math:`[-scale, scale]`.

    Attributes:
        scale (float): A constant that determines the
            scale of the uniform distribution.
        dtype: Data type specifier.
        rng (xp.random.RandomState): Pseudo-random number generator.

    """

    def __init__(self, scale=0.05, dtype=None, **kwargs):
        self.scale = scale
        rng = None
        if kwargs:
            rng, = argument.parse_kwargs(kwargs, ('rng', rng))
        self.rng = rng
        super(Uniform, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype,\
                '{} != {}'.format(array.dtype, self.dtype)
        if self.rng is None:
            device = backend.get_device_from_array(array)
            array[...] = device.xp.random.uniform(
                low=-self.scale, high=self.scale, size=array.shape)
        else:
            backend.copyto(array, self.rng.uniform(
                low=-self.scale, high=self.scale,
                size=array.shape).astype(array.dtype, copy=False))


class LeCunUniform(initializer.Initializer):

    """Initializes array with a scaled uniform distribution.

    Each element of the array is initialized by the value drawn
    independently from uniform distribution :math:`[-s, s]`
    where :math:`s = scale \\times \\sqrt{\\frac{3}{fan_{in}}}`.
    Here :math:`fan_{in}` is the number of input units.

    Reference: LeCun 98, Efficient Backprop
    http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

    Attributes:
        scale (float): A constant that determines the
            scale of the uniform distribution.
        dtype: Data type specifier.
        rng (xp.random.RandomState): Pseudo-random number generator.

    """

    def __init__(self, scale=1.0, dtype=None, **kwargs):
        self.scale = scale
        rng = None
        if kwargs:
            rng, = argument.parse_kwargs(kwargs, ('rng', rng))
        self.rng = rng
        super(LeCunUniform, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype,\
                '{} != {}'.format(array.dtype, self.dtype)
        fan_in, fan_out = initializer.get_fans(array.shape)
        s = self.scale * numpy.sqrt(3. / fan_in)
        Uniform(s, rng=self.rng)(array)


class GlorotUniform(initializer.Initializer):

    """Initializes array with a scaled uniform distribution.

    Each element of the array is initialized by the value drawn
    independently from uniform distribution :math:`[-s, s]`
    where :math:`s = scale \\times \\sqrt{\\frac{6}{fan_{in} + fan_{out}}}`.
    Here, :math:`fan_{in}` and :math:`fan_{out}` are the number of
    input and output units, respectively.

    Attributes:
        scale (float): A constant that determines the
            scale of the uniform distribution.
        dtype: Data type specifier.
        rng (xp.random.RandomState): Pseudo-random number generator.

    """

    def __init__(self, scale=1.0, dtype=None, **kwargs):
        self.scale = scale
        rng = None
        if kwargs:
            rng, = argument.parse_kwargs(kwargs, ('rng', rng))
        self.rng = rng
        super(GlorotUniform, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype,\
                '{} != {}'.format(array.dtype, self.dtype)
        fan_in, fan_out = initializer.get_fans(array.shape)
        s = self.scale * numpy.sqrt(6. / (fan_in + fan_out))
        Uniform(s, rng=self.rng)(array)


class HeUniform(initializer.Initializer):

    """Initializes array with scaled uniform distribution.

    Each element of the array is initialized by the value drawn
    independently from uniform distribution :math:`[-s, s]`
    where :math:`s = scale \\times \\sqrt{\\frac{6}{fan_{in}}}`.
    Here, :math:`fan_{in}` is the number of input units.

    Attributes:
        scale (float): A constant that determines the
            scale of the uniform distribution.
        dtype: Data type specifier.
        rng (xp.random.RandomState): Pseudo-random number generator.

    """

    def __init__(self, scale=1.0, dtype=None, **kwargs):
        self.scale = scale
        rng = None
        if kwargs:
            rng, = argument.parse_kwargs(kwargs, ('rng', rng))
        self.rng = rng
        super(HeUniform, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype,\
                '{} != {}'.format(array.dtype, self.dtype)
        fan_in, fan_out = initializer.get_fans(array.shape)
        s = self.scale * numpy.sqrt(6. / fan_in)
        Uniform(s, rng=self.rng)(array)
