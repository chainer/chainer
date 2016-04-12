import numpy

import chainer
from chainer import cuda
from chainer.functions.activation import hard_sigmoid
from chainer.functions.activation import sigmoid
from chainer.functions.activation import softplus
from chainer.functions.activation import tanh
from chainer import link
from chainer.links.connection import linear
from chainer import variable


class SGU(link.Chain):

    def __init__(self, in_size, out_size):
        super(SGU, self).__init__(
            W_xh=linear.Linear(in_size, out_size),
            W_zxh=linear.Linear(out_size, out_size),
            W_xz=linear.Linear(in_size, out_size),
            W_hz=linear.Linear(out_size, out_size),
        )

    def __call__(self, h, x):
        x_g = self.W_xh(x)
        z_g = tanh.tanh(self.W_zxh(x_g * h))
        z_out = softplus.softplus(z_g * h)
        z_t = hard_sigmoid.hard_sigmoid(self.W_xz(x) + self.W_hz(h))
        h_t = (1 - z_t) * h + z_t * z_out
        return h_t


class StatefulSGU(SGU):

    def __init__(self, in_size, out_size):
        super(StatefulSGU, self).__init__(in_size, out_size)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(StatefulSGU, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(StatefulSGU, self).to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h):
        assert isinstance(h, chainer.Variable)
        h_ = h
        if self.xp == numpy:
            h_.to_cpu()
        else:
            h_.to_gpu()
        self.h = h_

    def reset_state(self):
        self.h = None

    def __call__(self, x):

        if self.h is None:
            xp = cuda.get_array_module(x)
            zero = variable.Variable(xp.zeros_like(x.data))
            z_out = softplus.softplus(zero)
            z_t = hard_sigmoid.hard_sigmoid(self.W_xz(x))
            h_t = z_t * z_out
        else:
            h_t = SGU.__call__(self, self.h, x)

        self.h = h_t
        return h_t


class DSGU(link.Chain):

    def __init__(self, in_size, out_size):
        super(DSGU, self).__init__(
            W_xh=linear.Linear(in_size, out_size),
            W_zxh=linear.Linear(out_size, out_size),
            W_go=linear.Linear(out_size, out_size),
            W_xz=linear.Linear(in_size, out_size),
            W_hz=linear.Linear(out_size, out_size),
        )

    def __call__(self, h, x):
        x_g = self.W_xh(x)
        z_g = tanh.tanh(self.W_zxh(x_g * h))
        z_out = sigmoid.sigmoid(self.W_go(z_g * h))
        z_t = hard_sigmoid.hard_sigmoid(self.W_xz(x) + self.W_hz(h))
        h_t = (1 - z_t) * h + z_t * z_out
        return h_t


class StatefulDSGU(DSGU):

    def __init__(self, in_size, out_size):
        super(StatefulDSGU, self).__init__(in_size, out_size)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(StatefulDSGU, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(StatefulDSGU, self).to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h):
        assert isinstance(h, chainer.Variable)
        h_ = h
        if self.xp == numpy:
            h_.to_cpu()
        else:
            h_.to_gpu()
        self.h = h_

    def reset_state(self):
        self.h = None

    def __call__(self, x):

        if self.h is None:
            z_t = hard_sigmoid.hard_sigmoid(self.W_xz(x))
            h_t = z_t * 0.5
        else:
            h_t = DSGU.__call__(self, self.h, x)

        self.h = h_t
        return h_t
