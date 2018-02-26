import numpy

from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.math import linear_interpolate
from chainer import link
from chainer.links.connection import linear
from chainer import variable


class GRUBase(link.Chain):

    def __init__(self, in_size, out_size, init=None,
                 inner_init=None, bias_init=None):
        super(GRUBase, self).__init__()
        with self.init_scope():
            self.W_r = linear.Linear(
                in_size, out_size, initialW=init, initial_bias=bias_init)
            self.U_r = linear.Linear(
                out_size, out_size, initialW=inner_init,
                initial_bias=bias_init)
            self.W_z = linear.Linear(
                in_size, out_size, initialW=init, initial_bias=bias_init)
            self.U_z = linear.Linear(
                out_size, out_size, initialW=inner_init,
                initial_bias=bias_init)
            self.W = linear.Linear(
                in_size, out_size, initialW=init, initial_bias=bias_init)
            self.U = linear.Linear(
                out_size, out_size, initialW=inner_init,
                initial_bias=bias_init)


class StatelessGRU(GRUBase):

    """Stateless Gated Recurrent Unit function (GRU).

    GRU function has six parameters :math:`W_r`, :math:`W_z`, :math:`W`,
    :math:`U_r`, :math:`U_z`, and :math:`U`.
    The three parameters :math:`W_r`, :math:`W_z`, and :math:`W` are
    :math:`n \\times m` matrices, and the others :math:`U_r`, :math:`U_z`,
    and :math:`U` are :math:`n \\times n` matrices, where :math:`m` is the
    length of input vectors and :math:`n` is the length of hidden vectors.

    Given two inputs a previous hidden vector :math:`h` and an input vector
    :math:`x`, GRU returns the next hidden vector :math:`h'` defined as

    .. math::

       r &=& \\sigma(W_r x + U_r h), \\\\
       z &=& \\sigma(W_z x + U_z h), \\\\
       \\bar{h} &=& \\tanh(W x + U (r \\odot h)), \\\\
       h' &=& (1 - z) \\odot h + z \\odot \\bar{h},

    where :math:`\\sigma` is the sigmoid function, and :math:`\\odot` is the
    element-wise product.

    As the name indicates, :class:`~chainer.links.StatelessGRU` is *stateless*,
    meaning that it does not hold the value of
    hidden vector :math:`h`.
    For a *stateful* GRU, use :class:`~chainer.links.StatefulGRU`.

    Args:
        in_size(int): Dimension of input vector :math:`x`.
            If ``None``, parameter initialization will be deferred
            until the first forward data pass
            at which time the size will be determined.
        out_size(int): Dimension of hidden vector :math:`h`,
            :math:`\\bar{h}` and :math:`h'`.

    See:
        - `On the Properties of Neural Machine Translation: Encoder-Decoder
          Approaches <https://www.aclweb.org/anthology/W14-4012>`_
          [Cho+, SSST2014].
        - `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
          Modeling <https://arxiv.org/abs/1412.3555>`_
          [Chung+NIPS2014 DLWorkshop].


    .. seealso:: :class:`~chainer.links.StatefulGRU`

    .. admonition:: Example

        There are several ways to make a ``StatelessGRU`` link.
        Let ``x`` be a two-dimensional input array:

        >>> in_size = 10
        >>> out_size = 20
        >>> x = np.zeros((1, in_size), dtype=np.float32)
        >>> h = np.zeros((1, out_size), dtype=np.float32)

        1. Give both  ``in_size`` and ``out_size`` arguments:

            >>> l = L.StatelessGRU(in_size, out_size)
            >>> h_new = l(h, x)
            >>> h_new.shape
            (1, 20)

        2. Omit ``in_size`` argument or fill it with ``None``:

            >>> l = L.StatelessGRU(None, out_size)
            >>> h_new = l(h, x)
            >>> h_new.shape
            (1, 20)

    """

    def __call__(self, h, x):
        r = sigmoid.sigmoid(self.W_r(x) + self.U_r(h))
        z = sigmoid.sigmoid(self.W_z(x) + self.U_z(h))
        h_bar = tanh.tanh(self.W(x) + self.U(r * h))
        h_new = linear_interpolate.linear_interpolate(z, h_bar, h)
        return h_new


class StatefulGRU(GRUBase):
    """Stateful Gated Recurrent Unit function (GRU).

    Stateful GRU function has six parameters :math:`W_r`, :math:`W_z`,
    :math:`W`, :math:`U_r`, :math:`U_z`, and :math:`U`.
    The three parameters :math:`W_r`, :math:`W_z`, and :math:`W` are
    :math:`n \\times m` matrices, and the others :math:`U_r`, :math:`U_z`,
    and :math:`U` are :math:`n \\times n` matrices, where :math:`m` is the
    length of input vectors and :math:`n` is the length of hidden vectors.

    Given input vector :math:`x`, Stateful GRU returns the next
    hidden vector :math:`h'` defined as

    .. math::

       r &=& \\sigma(W_r x + U_r h), \\\\
       z &=& \\sigma(W_z x + U_z h), \\\\
       \\bar{h} &=& \\tanh(W x + U (r \\odot h)), \\\\
       h' &=& (1 - z) \\odot h + z \\odot \\bar{h},

    where :math:`h` is current hidden vector.

    As the name indicates, :class:`~chainer.links.StatefulGRU` is *stateful*,
    meaning that it also holds the next hidden vector `h'` as a state.
    For a *stateless* GRU, use :class:`~chainer.links.StatelessGRU`.

    Args:
        in_size(int): Dimension of input vector :math:`x`.
        out_size(int): Dimension of hidden vector :math:`h`.
        init: Initializer for GRU's input units (:math:`W`).
            It is a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            If it is ``None``, the default initializer is used.
        inner_init: Initializer for the GRU's inner
            recurrent units (:math:`U`).
            It is a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            If it is ``None``, the default initializer is used.
        bias_init: Bias initializer.
            It is a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            If ``None``, the bias is set to zero.

    Attributes:
        h(~chainer.Variable): Hidden vector that indicates the state of
            :class:`~chainer.links.StatefulGRU`.

    .. seealso::
        * :class:`~chainer.links.StatelessGRU`
        * :class:`~chainer.links.GRU`: an alias of
          :class:`~chainer.links.StatefulGRU`

    .. admonition:: Example

        There are several ways to make a ``StatefulGRU`` link.
        Let ``x`` be a two-dimensional input array:

        >>> in_size = 10
        >>> out_size = 20
        >>> x = np.zeros((1, in_size), dtype=np.float32)

        1. Give only ``in_size`` and ``out_size`` arguments:

            >>> l = L.StatefulGRU(in_size, out_size)
            >>> h_new = l(x)
            >>> h_new.shape
            (1, 20)

        2. Give all optional arguments:

            >>> init = np.zeros((out_size, in_size), dtype=np.float32)
            >>> inner_init = np.zeros((out_size, out_size), dtype=np.float32)
            >>> bias = np.zeros((1, out_size), dtype=np.float32)
            >>> l = L.StatefulGRU(in_size, out_size, init=init,
            ...     inner_init=inner_init, bias_init=bias)
            >>> h_new = l(x)
            >>> h_new.shape
            (1, 20)

    """

    def __init__(self, in_size, out_size, init=None,
                 inner_init=None, bias_init=0):
        super(StatefulGRU, self).__init__(
            in_size, out_size, init, inner_init, bias_init)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(StatefulGRU, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(StatefulGRU, self).to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h):
        assert isinstance(h, variable.Variable)
        h_ = h
        if self.xp == numpy:
            h_.to_cpu()
        else:
            h_.to_gpu(self._device_id)
        self.h = h_

    def reset_state(self):
        self.h = None

    def __call__(self, x):
        z = self.W_z(x)
        h_bar = self.W(x)
        if self.h is not None:
            r = sigmoid.sigmoid(self.W_r(x) + self.U_r(self.h))
            z += self.U_z(self.h)
            h_bar += self.U(r * self.h)
        z = sigmoid.sigmoid(z)
        h_bar = tanh.tanh(h_bar)

        if self.h is not None:
            h_new = linear_interpolate.linear_interpolate(z, h_bar, self.h)
        else:
            h_new = z * h_bar
        self.h = h_new
        return self.h


class GRU(StatefulGRU):
    """Stateful Gated Recurrent Unit function (GRU)

    This is an alias of :class:`~chainer.links.StatefulGRU`.

    .. warning::

       In Chainer v1, ``GRU`` was *stateless*,
       as opposed to the current implementation.
       To align with LSTM links, we have changed
       the naming convention from Chainer v2 so that the shorthand name
       points the stateful links.
       You can use :class:`~chainer.links.StatelessGRU` for stateless version,
       whose implementation is identical to ``GRU`` in v1.

       See issue `#2537 <https://github.com/chainer/chainer/issues/2537>`_
       for details.

    """

    def __call__(self, *args):
        """__call__(self, x)

        Does forward propagation.

        """

        n_args = len(args)
        msg = ("Invalid argument. The length of GRU.__call__ must be 1. "
               "But %d is given. " % n_args)

        if n_args == 0 or n_args >= 3:
            raise ValueError(msg)
        elif n_args == 2:
            msg += ("In Chainer v2, chainer.links.GRU is changed "
                    "from stateless to stateful. "
                    "One possiblity is you assume GRU to be stateless. "
                    "Use chainer.links.StatelessGRU instead.")
            raise ValueError(msg)

        return super(GRU, self).__call__(args[0])
