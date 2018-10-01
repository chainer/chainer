import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import argument
from chainer.utils import type_check
import numpy as np


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

    def forward_cpu(self, x):
        if chainer.config.train:
            if self.r is None:
                self.r = np.random.uniform(
                    self.lower, self.upper, x[0].shape).astype(x[0].dtype)
        else:
            self.r = np.full(
                x[0].shape, (self.lower + self.upper) / 2).astype(x[0].dtype)
        y = np.where(x[0] >= 0, x[0], x[0] * self.r)
        self.retain_outputs((0,))
        return y,

    def forward_gpu(self, x):
        xp = cuda.cupy
        if chainer.config.train:
            if self.r is None:
                self.r = xp.random.uniform(
                    self.lower, self.upper, x[0].shape).astype(x[0].dtype)
        else:
            self.r = xp.full(
                x[0].shape, (self.lower + self.upper) / 2).astype(x[0].dtype)
        y = _kern()(x[0], x[0], self.r.astype(x[0].dtype))
        self.retain_inputs(())
        self.retain_outputs((0,))
        return y,

    def backward(self, indexes, grad_outputs):
        x = None
        y = self.get_retained_outputs()[0].data
        return _RReLUGrad(x, y, self.r).apply(grad_outputs)


class _RReLUGrad(function_node.FunctionNode):

    def __init__(self, x, y, r):
        self.r = r
        self.x = x
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
        return _RReLUGrad(self.x, self.y, self.r).apply(grad_outputs)


def rrelu(x, l=1. / 8, u=1. / 3, **kwargs):
    """rrelu(x, l=1. / 8, u=1. / 3, *, r=None, return_r=False)

        Randomized Leaky Rectified Liner Unit function.

    This function is expressed as

    .. math:: f(x)=\\max(x, ax),

    where :math:`a` is a random number sampled \
                from a uniform distribution :math:`U(l, u)`.

    See: https://arxiv.org/pdf/1505.00853.pdf

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        l (float): The lower bound of the uniform distribution.
        u (float): The upper bound of the uniform distribution.
        r (:class:`numpy.ndarray` or None):
            The r to be used for rrelu.
            The shape and dtype must be the same as ``x[0]`` and should be on
            the same device.
            If ``r``  is not specified or set to ``None``, a ``r`` will be
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
            ``r`` (ndarray). The ``r`` will be on the same device as the input.
            A :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    .. admonition:: Example

        >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], 'f')
        >>> x
        array([[-1.,  0.],
               [ 2., -3.],
               [-2.,  1.]], dtype=float32)
        >>> F.rrelu(x).data # doctest: +SKIP
        array([[-0.24850948,  0.        ],
               [ 2.        , -0.50844127],
               [-0.598535  ,  1.        ]], dtype=float32)
    """
    r = None
    return_r = False
    if kwargs:
        r, return_r = argument.parse_kwargs(
            kwargs, ('r', r), ('return_r', r),
            train='train argument is not supported anymore.'
                  'Use chainer.using_config')

    func = RReLU(l, u, r)
    out = func.apply((x,))[0]
    r = func.r

    if return_r:
        return out, r
    return out
