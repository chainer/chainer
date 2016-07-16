import numpy

import chainer
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.array import concat
from chainer.functions.math import linear_interpolate
from chainer import link
from chainer.links.connection import linear


class MGU(link.Chain):

    def __init__(self, n_inputs, n_units):
        super(MGU, self).__init__(
            W_f=linear.Linear(n_inputs + n_units, n_units),
            W_h=linear.Linear(n_inputs + n_units, n_units)
        )

    def __call__(self, h, x):
        f = sigmoid.sigmoid(self.W_f(concat.concat([h, x])))
        h_bar = tanh.tanh(self.W_h(concat.concat([f * h, x])))
        h_new = linear_interpolate.linear_interpolate(f, h_bar, h)
        return h_new


class StatefulMGU(MGU):

    def __init__(self, in_size, out_size):
        super(StatefulMGU, self).__init__(in_size, out_size)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(StatefulMGU, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(StatefulMGU, self).to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def set_state(self, h):
        assert isinstance(h, chainer.Variable)
        h_ = h
        if self.xp is numpy:
            h_.to_cpu()
        else:
            h_.to_gpu()
        self.h = h_

    def reset_state(self):
        self.h = None

    def __call__(self, x):
        if self.h is None:
            n_batch = len(x.data)
            h_data = self.xp.zeros(
                (n_batch, self.state_size), dtype=numpy.float32)
            h = chainer.Variable(h_data)
        else:
            h = self.h

        self.h = MGU.__call__(self, h, x)
        return self.h
