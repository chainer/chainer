import numpy as np

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import argument
from chainer.utils import type_check


def _kern():
    return cuda.elementwise(
        'T cond, T x, T slope', 'T y',
        'y = cond >= 0 ? x : (T)(slope * x)', 'rrelu')


class RReLU(function_node.FunctionNode):
    """Randomized Leaky rectifier unit."""

    def __init__(self, lower=1. / 8, upper=1. / 3, r=None):
        if not 0.0 <= lower < 1.0:
            raise ValueError('lower must be in the range [0, 1)')
        if not 0.0 <= upper < 1.0:
            raise ValueError('upper must be in the range [0, 1)')
        if not lower < upper:
            raise ValueError('lower must be less than upper')
        self.lower = lower
        self.upper = upper
        self.r = r

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types
        type_check.expect(x_type.dtype.kind == 'f')
        if self.r is not None:
            type_check.expect(x_type.dtype == self.r.dtype)
            type_check.expect(x_type.shape == self.r.shape)

    def forward_cpu(self, inputs):
        x, = inputs
        if chainer.config.train:
            if self.r is None:
                self.r = np.random.uniform(
                    self.lower, self.upper, x.shape
                ).astype(x.dtype, copy=False)
        else:
            self.r = np.full(
                x.shape, (self.lower + self.upper) / 2, dtype=x.dtype)
        y = np.where(x >= 0, x, x * self.r)
        self.retain_outputs((0,))
        return y,

    def forward_gpu(self, inputs):
        x, = inputs
        xp = cuda.cupy
        if chainer.config.train:
            if self.r is None:
                self.r = xp.random.uniform(
                    self.lower, self.upper, x.shape
                ).astype(x.dtype, copy=False)
        else:
            self.r = xp.full(
                x.shape, (self.lower + self.upper) / 2, dtype=x.dtype)
        y = _kern()(x, x, self.r)
        self.retain_outputs((0,))
        return y,

    def backward(self, indexes, grad_outputs):
        y = self.get_retained_outputs()[0].data
        return _RReLUGrad(y, self.r).apply(grad_outputs)


class _RReLUGrad(function_node.FunctionNode):

    def __init__(self, y, r):
        self.r = r
        self.y = y

    def forward_cpu(self, inputs):
        gy, = inputs
        gy = np.where(self.y >= 0, gy, gy * self.r)
        return gy,

    def forward_gpu(self, inputs):
        gy, = inputs
        gy = _kern()(self.y, gy, self.r)
        return gy,

    def backward(self, indexes, grad_outputs):
        return _RReLUGrad(self.y, self.r).apply(grad_outputs)


def rrelu(x, l=1. / 8, u=1. / 3, **kwargs):
    """rrelu(x, l=1. / 8, u=1. / 3, *, r=None, return_r=False)

    Randomized Leaky Rectified Liner Unit function.

    This function is expressed as

    .. math:: f(x)=\\max(x, rx),

    where :math:`r` is a random number sampled from a uniform distribution
    :math:`U(l, u)`.

    .. note::

        The :math:`r` corresponds to :math:`a` in the original
        paper (https://arxiv.org/pdf/1505.00853.pdf).

    Args:
        x (:class:`~chainer.Variable` or :ref:`ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        l (float): The lower bound of the uniform distribution.
        u (float): The upper bound of the uniform distribution.
        r (:ref:`ndarray` or None):
            The r to be used for rrelu.
            The shape and dtype must be the same as ``x[0]`` and should be on
            the same device.
            If ``r``  is not specified or set to ``None``, an ``r`` will be
            generated randomly according to the given ``l`` and ``u``.
            If ``r`` is specified, ``l`` and ``u`` will be ignored.
        return_r (bool):
            If ``True``, the r used for rrelu is returned altogether with
            the output variable.
            The returned ``r`` can latter be reused by passing it to ``r``
            argument.

    Returns:
        ~chainer.Variable or tuple:
            When ``return_r`` is ``False`` (default), return the output
            variable. Otherwise returnes the tuple of the output variable and
            ``r`` (:ref:`ndarray`). The ``r`` will be on the same device as
            the input.
            A :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    .. admonition:: Example

        >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
        >>> x
        array([[-1.,  0.],
               [ 2., -3.],
               [-2.,  1.]], dtype=float32)
        >>> F.rrelu(x).array # doctest: +SKIP
        array([[-0.24850948,  0.        ],
               [ 2.        , -0.50844127],
               [-0.598535  ,  1.        ]], dtype=float32)
    """
    r = None
    return_r = False
    if kwargs:
        r, return_r = argument.parse_kwargs(
            kwargs, ('r', r), ('return_r', r),
            train='train argument is not supported anymore. '
                  'Use chainer.using_config')

    func = RReLU(l, u, r)
    out, = func.apply((x,))
    r = func.r

    if return_r:
        return out, r
    return out
