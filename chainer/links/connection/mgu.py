from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.array import concat
from chainer.functions.math import linear_interpolate
from chainer import link
from chainer.links.connection import linear
from chainer.utils import rnn


class MGU(link.Chain):

    state_names = 'h',

    def __init__(self, n_inputs, n_units):
        super(MGU, self).__init__(
            W_f=linear.Linear(n_inputs + n_units, n_units),
            W_h=linear.Linear(n_inputs + n_units, n_units)
        )
        self.state_shapes = (n_units,),

    def __call__(self, h, x):
        f = sigmoid.sigmoid(self.W_f(concat.concat([h, x])))
        h_bar = tanh.tanh(self.W_h(concat.concat([f * h, x])))
        h_new = linear_interpolate.linear_interpolate(f, h_bar, h)
        return h_new


StatefulMGU = rnn.create_stateful_rnn(MGU, 'StatefulMGU')
