import numpy

import chainer
from chainer.backends import cuda
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.array import concat
from chainer.functions.math import linear_interpolate
from chainer import link
from chainer.links.connection import linear


class MGUBase(link.Chain):

    def __init__(self, n_inputs, n_units):
        super(MGUBase, self).__init__()
        with self.init_scope():
            self.W_f = linear.Linear(n_inputs + n_units, n_units)
            self.W_h = linear.Linear(n_inputs + n_units, n_units)

    def _call_mgu(self, h, x):
        f = sigmoid.sigmoid(self.W_f(concat.concat([h, x])))
        h_bar = tanh.tanh(self.W_h(concat.concat([f * h, x])))
        h_new = linear_interpolate.linear_interpolate(f, h_bar, h)
        return h_new


class StatelessMGU(MGUBase):

    forward = MGUBase._call_mgu


class StatefulMGU(MGUBase):

    def __init__(self, in_size, out_size):
        super(StatefulMGU, self).__init__(in_size, out_size)
        self._state_size = out_size
        self.reset_state()

    def _to_device(self, device, skip_between_cupy_devices=False):
        # Overrides Link._to_device
        # TODO(niboshi): Avoid forcing concrete links to override _to_device
        device = chainer.get_device(device)
        super(StatefulMGU, self)._to_device(
            device, skip_between_cupy_devices=skip_between_cupy_devices)
        if self.h is not None:
            if not (skip_between_cupy_devices
                    and device.xp is cuda.cupy
                    and isinstance(self.h, cuda.ndarray)):
                self.h.to_device(device)
        return self

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

    def forward(self, x):
        if self.h is None:
            n_batch = x.shape[0]
            dtype = chainer.get_dtype()
            h_data = self.xp.zeros(
                (n_batch, self._state_size), dtype=dtype)
            h = chainer.Variable(h_data)
        else:
            h = self.h

        self.h = self._call_mgu(h, x)
        return self.h
