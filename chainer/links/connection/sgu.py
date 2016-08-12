from chainer.functions.activation import hard_sigmoid
from chainer.functions.activation import sigmoid
from chainer.functions.activation import softplus
from chainer.functions.activation import tanh
from chainer.functions.math import linear_interpolate
from chainer import link
from chainer.links.connection import linear
from chainer.utils import rnn


class SGU(link.Chain):

    state_names = 'h',

    def __init__(self, in_size, out_size):
        super(SGU, self).__init__(
            W_xh=linear.Linear(in_size, out_size),
            W_zxh=linear.Linear(out_size, out_size),
            W_xz=linear.Linear(in_size, out_size),
            W_hz=linear.Linear(out_size, out_size),
        )
        self.state_shapes = (out_size,),

    def __call__(self, h, x):
        x_g = self.W_xh(x)
        z_g = tanh.tanh(self.W_zxh(x_g * h))
        z_out = softplus.softplus(z_g * h)
        z_t = hard_sigmoid.hard_sigmoid(self.W_xz(x) + self.W_hz(h))
        h_t = linear_interpolate.linear_interpolate(z_t, z_out, h)
        return h_t


StatefulSGU = rnn.create_stateful_rnn(SGU, "StatefulSGU")


class DSGU(link.Chain):

    state_names = 'h'

    def __init__(self, in_size, out_size):
        super(DSGU, self).__init__(
            W_xh=linear.Linear(in_size, out_size),
            W_zxh=linear.Linear(out_size, out_size),
            W_go=linear.Linear(out_size, out_size),
            W_xz=linear.Linear(in_size, out_size),
            W_hz=linear.Linear(out_size, out_size),
        )
        self.state_shapes = (out_size,),

    def __call__(self, h, x):
        x_g = self.W_xh(x)
        z_g = tanh.tanh(self.W_zxh(x_g * h))
        z_out = sigmoid.sigmoid(self.W_go(z_g * h))
        z_t = hard_sigmoid.hard_sigmoid(self.W_xz(x) + self.W_hz(h))
        h_t = linear_interpolate.linear_interpolate(z_t, z_out, h)
        return h_t


StatefulDSGU = rnn.create_stateful_rnn(DSGU, "StatefulDSGU")
