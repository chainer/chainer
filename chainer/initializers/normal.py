import numpy

from chainer import backend
from chainer.backends import cuda
from chainer import initializer
from chainer.utils import argument


# Original code forked from MIT licensed keras project
# https://github.com/fchollet/keras/blob/master/keras/initializations.py

class Normal(initializer.Initializer):

    """Initializes array with a normal distribution.

    Each element of the array is initialized by the value drawn
    independently from Gaussian distribution whose mean is 0,
    and standard deviation is ``scale``.

    Args:
        scale (float): Standard deviation of Gaussian distribution.
        dtype: Data type specifier.
        rng (xp.random.RandomState): Pseudo-random number generator.

    """

    def __init__(self, scale=0.05, dtype=None, **kwargs):
        self.scale = scale
        rng = None
        if kwargs:
            rng, = argument.parse_kwargs(kwargs, ('rng', rng))
        self.rng = rng
        super(Normal, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype,\
                '{} != {}'.format(array.dtype, self.dtype)
        if self.rng is None:
            device = backend.get_device_from_array(array)
            args = {'loc': 0.0, 'scale': self.scale, 'size': array.shape}
            if device.xp is cuda.cupy:
                # Only CuPy supports dtype option
                if self.dtype == numpy.float32 or self.dtype == numpy.float16:
                    # float16 is not supported in cuRAND
                    args['dtype'] = numpy.float32
            array[...] = device.xp.random.normal(**args)
        else:
            backend.copyto(array, self.rng.normal(
                loc=0.0, scale=self.scale,
                size=array.shape).astype(array.dtype, copy=False))


class LeCunNormal(initializer.Initializer):

    """Initializes array with scaled Gaussian distribution.

    Each element of the array is initialized by the value drawn
    independently from Gaussian distribution whose mean is 0,
    and standard deviation is
    :math:`scale \\times \\sqrt{\\frac{1}{fan_{in}}}`,
    where :math:`fan_{in}` is the number of input units.

    Reference: LeCun 98, Efficient Backprop
    http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf

    Args:
        scale (float): A constant that determines the scale
            of the standard deviation.
        dtype: Data type specifier.
        rng (xp.random.RandomState): Pseudo-random number generator.

    """

    def __init__(self, scale=1.0, dtype=None, **kwargs):
        self.scale = scale
        rng = None
        if kwargs:
            rng, = argument.parse_kwargs(kwargs, ('rng', rng))
        self.rng = rng
        super(LeCunNormal, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype,\
                '{} != {}'.format(array.dtype, self.dtype)
        fan_in, fan_out = initializer.get_fans(array.shape)
        s = self.scale * numpy.sqrt(1. / fan_in)
        Normal(s, rng=self.rng)(array)


class GlorotNormal(initializer.Initializer):

    """Initializes array with scaled Gaussian distribution.

    Each element of the array is initialized by the value drawn
    independently from Gaussian distribution whose mean is 0,
    and standard deviation is
    :math:`scale \\times \\sqrt{\\frac{2}{fan_{in} + fan_{out}}}`,
    where :math:`fan_{in}` and :math:`fan_{out}` are the number of
    input and output units, respectively.

    Reference: Glorot & Bengio, AISTATS 2010

    Args:
        scale (float): A constant that determines the scale
            of the standard deviation.
        dtype: Data type specifier.
        rng (xp.random.RandomState): Pseudo-random number generator.

    """

    def __init__(self, scale=1.0, dtype=None, **kwargs):
        self.scale = scale
        rng = None
        if kwargs:
            rng, = argument.parse_kwargs(kwargs, ('rng', rng))
        self.rng = rng
        super(GlorotNormal, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype,\
                '{} != {}'.format(array.dtype, self.dtype)
        fan_in, fan_out = initializer.get_fans(array.shape)
        s = self.scale * numpy.sqrt(2. / (fan_in + fan_out))
        Normal(s, rng=self.rng)(array)


class HeNormal(initializer.Initializer):

    """Initializes array with scaled Gaussian distribution.

    Each element of the array is initialized by the value drawn
    independently from Gaussian distribution whose mean is 0,
    and standard deviation is
    :math:`scale \\times \\sqrt{\\frac{2}{fan}}`.
    If ``fan_option == 'fan_in'``, :math:`fan` is the
    number of input units.
    If ``fan_option == 'fan_out'``, :math:`fan` is the
    number of output units.

    Reference:  He et al., https://arxiv.org/abs/1502.01852

    Args:
        scale (float): A constant that determines the scale
            of the standard deviation.
        dtype: Data type specifier.
        fan_option ({'fan_in', 'fan_out'}): Decides how to compute the
            standard deviation. The default value is ``'fan_in'``.
        rng (xp.random.RandomState): Pseudo-random number generator.

    """

    def __init__(self, scale=1.0, dtype=None, fan_option='fan_in', **kwargs):
        self.scale = scale
        self.fan_option = fan_option
        rng = None
        if kwargs:
            rng, = argument.parse_kwargs(kwargs, ('rng', rng))
        self.rng = rng
        super(HeNormal, self).__init__(dtype)

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype,\
                '{} != {}'.format(array.dtype, self.dtype)
        fan_in, fan_out = initializer.get_fans(array.shape)
        if self.fan_option == 'fan_in':
            s = self.scale * numpy.sqrt(2. / fan_in)
        elif self.fan_option == 'fan_out':
            s = self.scale * numpy.sqrt(2. / fan_out)
        else:
            raise ValueError(
                'fan_option should be either \'fan_in\' or \'fan_out\'.')
        Normal(s, rng=self.rng)(array)
