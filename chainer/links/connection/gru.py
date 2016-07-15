import numpy

import chainer
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer import link
from chainer.links.connection import linear
from chainer.utils import rnn


class GRUBase(link.Chain):

    state_names = 'h',

    def __init__(self, n_units, n_inputs=None, init=None,
                 inner_init=None, bias_init=0):
        self.state_shapes = (n_units,),
        if n_inputs is None:
            n_inputs = n_units
        super(GRUBase, self).__init__(
            W_r=linear.Linear(n_inputs, n_units,
                              initialW=init, initial_bias=bias_init),
            U_r=linear.Linear(n_units, n_units,
                              initialW=inner_init, initial_bias=bias_init),
            W_z=linear.Linear(n_inputs, n_units,
                              initialW=init, initial_bias=bias_init),
            U_z=linear.Linear(n_units, n_units,
                              initialW=inner_init, initial_bias=bias_init),
            W=linear.Linear(n_inputs, n_units,
                            initialW=init, initial_bias=bias_init),
            U=linear.Linear(n_units, n_units,
                            initialW=inner_init, initial_bias=bias_init),
        )


class GRU(GRUBase):

    """Stateless Gated Recurrent Unit function (GRU).

    GRU function has six parameters :math:`W_r`, :math:`W_z`, :math:`W`,
    :math:`U_r`, :math:`U_z`, and :math:`U`. All these parameters are
    :math:`n \\times n` matrices, where :math:`n` is the dimension of
    hidden vectors.

    Given two inputs a previous hidden vector :math:`h` and an input vector
    :math:`x`, GRU returns the next hidden vector :math:`h'` defined as

    .. math::

       r &=& \\sigma(W_r x + U_r h), \\\\
       z &=& \\sigma(W_z x + U_z h), \\\\
       \\bar{h} &=& \\tanh(W x + U (r \\odot h)), \\\\
       h' &=& (1 - z) \\odot h + z \\odot \\bar{h},

    where :math:`\\sigma` is the sigmoid function, and :math:`\\odot` is the
    element-wise product.

    :class:`~chainer.links.GRU` does not hold the value of
    hidden vector :math:`h`. So this is *stateless*.
    Use :class:`~chainer.links.StatefulGRU` as a *stateful* GRU.

    Args:
        n_units(int): Dimension of hidden vector :math:`h`.
        n_inputs(int): Dimension of input vector :math:`x`. If ``None``,
            it is set to the same value as ``n_units``.

    See:
        - `On the Properties of Neural Machine Translation: Encoder-Decoder
          Approaches <http://www.aclweb.org/anthology/W14-4012>`_
          [Cho+, SSST2014].
        - `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
          Modeling <http://arxiv.org/abs/1412.3555>`_
          [Chung+NIPS2014 DLWorkshop].


    .. seealso:: :class:`~chainer.links.StatefulGRU`

    """

    def __call__(self, h, x):
        r = sigmoid.sigmoid(self.W_r(x) + self.U_r(h))
        z = sigmoid.sigmoid(self.W_z(x) + self.U_z(h))
        h_bar = tanh.tanh(self.W(x) + self.U(r * h))
        h_new = (1 - z) * h + z * h_bar
        return h_new


StatefulGRUBase = rnn.create_stateful_rnn(GRU, 'StatefulGRUBase')

class StatefulGRU(StatefulGRUBase):
    """Stateful Gated Recurrent Unit function (GRU).

    Stateful GRU function has six parameters :math:`W_r`, :math:`W_z`,
    :math:`W`, :math:`U_r`, :math:`U_z`, and :math:`U`.
    All these parameters are :math:`n \\times n` matrices,
    where :math:`n` is the dimension of hidden vectors.

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
    Use :class:`~chainer.links.GRU` as a stateless version of GRU.

    Args:
        in_size(int): Dimension of input vector :math:`x`.
        out_size(int): Dimension of hidden vector :math:`h`.
        init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the
            GRU's input units (:math:`W`). Maybe be `None` to use default
            initialization.
        inner_init: A callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
            It is used for initialization of the GRU's inner
            recurrent units (:math:`U`).
            Maybe be ``None`` to use default initialization.
        bias_init: A callable or scalar used to initialize the bias values for
            both the GRU's inner and input units. Maybe be ``None`` to use
            default initialization.

    Attributes:
        h(~chainer.Variable): Hidden vector that indicates the state of
            :class:`~chainer.links.StatefulGRU`.

    .. seealso:: :class:`~chainer.functions.GRU`

    """

    def __init__(self, in_size, out_size, **kwargs):
        super(StatefulGRU, self).__init__(
            out_size, in_size, **kwargs)

    def set_state(self, h):
        super(StatefulGRU, self).set_state(h)
