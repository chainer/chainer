import numpy

from chainer import functions as F
from chainer.backends import cuda
from chainer.function_node import FunctionNode
from chainer.utils import type_check


class HingeMaxMargin(FunctionNode):

    """Hinge max margin loss."""

    def __init__(self, norm='L2', reduce='mean'):
        if norm in ['L1', 'L2', 'Huber']:
            self.norm = norm
        else:
            raise NotImplementedError(
                "norm should be either 'L1', 'L2' or 'Huber'")

        if reduce in ['mean', 'along_second_axis']:
            self.reduce = reduce
        else:
            raise ValueError(
                "only 'mean' and 'along_second_axis' are valid for 'reduce',"
                " but '%s' is " 'given' % reduce)
        self.mask = None
        self.margin = None

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, t_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            t_type.dtype.kind == 'i',
            x_type.ndim == 2,
            t_type.ndim == 1,
            x_type.shape[0] == t_type.shape[0],
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, t = inputs

        num = len(x)
        self.mask = xp.zeros_like(x)
        self.mask[xp.arange(num), t] = -1
        temp = xp.copy(x)
        temp[xp.arange(num), t] = numpy.finfo(numpy.float32).min
        self.mask[xp.arange(num), xp.argmax(temp, 1)] = 1
        self.margin = xp.maximum(0, 1 + numpy.sum(self.mask * x, 1))

        if self.norm == 'L1':
            loss = self.margin
        elif self.norm == 'L2':
            loss = self.margin ** 2
        elif self.norm == 'Huber':
            quad = (self.margin < 2).astype(x.dtype)
            loss = self.margin**2 / 4 * quad + (self.margin-1)*(1-quad)
        else:
            raise NotImplementedError()

        if self.reduce == 'mean':
            loss = xp.array(loss.sum() / num, dtype=x.dtype)

        return loss,

    def backward(self, indexes, grad_outputs):
        gloss, = grad_outputs

        if self.reduce == 'mean':
            gloss /= self.margin.shape[0]

        if self.norm == 'L1':
            gx = gloss * F.sign(self.mask * F.expand_dims(self.margin, 1))
        elif self.norm == 'L2':
            gx = 2 * gloss * self.mask * F.expand_dims(self.margin, 1)
        elif self.norm == 'Huber':
            gx = gloss * self.mask * \
                F.expand_dims(F.minimum(
                    self.margin / 2, F.sign(self.margin)), 1)
        else:
            raise NotImplementedError()

        return gx, None


def hinge_max_margin(x, t, norm='L2', reduce='mean'):
    """Computes the hinge loss for a one vs max classification task.

        .. math::
            margin_{i} = ReLu \\left ( 1-x_{i,t_{i}}+max_{k,k\\neq t_{i}}
             \\left ( x_{i,k} \\right )\\right )

        and

        .. math::
            loss_{i} = \\left \\{ \\begin{array}{cc}
            margin_{i} & {\\rm if~norm} = {\\rm L1} \\\\
            margin_{i}^{2} & {\\rm if~norm} = {\\rm L2} \\\\
            margin_{i}-1 & \\rm if~norm} = {\\rm Huber \\& margin_{i}
             \\geqslant 2} \\\\
            margin_{i}^{2} & {\\rm if~norm} = {\\rm Huber \\& margin_{i}<2}

            \\end{array} \\right.

        All 3 norms are continuous.
        ``'L2'`` and ``'Huber'`` are differentiable, ``'L1'`` is not.

        The output is a variable whose value depends on the value of
        the option ``reduce``. If it is ``'along_second_axis'``,
         it holds the loss values for each example. If it is ``'mean'``,
          it takes the mean of loss values.

        See: `Huber loss- Wikipedia <https://en.wikipedia.org/wiki/Huber_loss>`
        and Structured support vector machine- Wikipedia
        <https://en.wikipedia.org/wiki/Structured_support_vector_machine>'

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray` of :class:`numpy.float`):
            Input variable. The shape of ``x`` should be (:math:`N`, :math:`K`)
            .
        t (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray` of signed integer):
            The :math:`N`-dimensional label vector with values
            :math:`t_n \\in \\{0, 1, 2, \\dots, K-1\\}`.
            The shape of ``t`` should be (:math:`N`,).
        norm (string): Specifies norm type. Either ``'L1'`` , ``'L2'`` ,
         ``'Huber'`` are acceptable.
        reduce (str): Reduction option. Its value must be either
            ``'mean'`` [default] or ``'along_second_axis'``.
             Otherwise, :class:`ValueError` is raised.

    Returns:
        ~chainer.Variable:
            A variable object holding the hinge max margin loss.
            If ``reduce`` is ``'along_second_axis'``, the output variable holds
             an array whose shape is same :math:`N`.
            If it is ``'mean'``, the output variable holds a scalar value.

    .. admonition:: Example

        In this case, the batch size ``N`` is 4 and the number of classes ``K``
        is 2.
        >>> import numpy as np
        >>> x = np.stack((np.arange(10),5*np.ones((10,))),1).astype(np.float32)
        >>> t = np.ones((10,),np.int32)
        >>> hinge_max_margin(x, t, norm='L1', reduce='along_second_axis')
        variable([0., 0., 0., 0., 0., 1., 2., 3., 4., 5.])
        >>> hinge_max_margin(x, t)
        variable(5.5)
        >>> hinge_max_margin(x, t, norm='L1')
        variable(1.5)
        >>> hinge_max_margin(x, t, norm='Huber')
        variable(1.025)


    """
    return HingeMaxMargin(norm, reduce).apply((x, t))[0]


if __name__ == '__main__':
    help(hinge_max_margin)
