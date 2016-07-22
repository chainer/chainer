import numpy

from chainer import link
from chainer import variable


def _force_tuple(x):
    if isinstance(x, variable.Variable):
        return (x,)
    else:
        return x


def create_stateful_rnn(stateless_class, name):

    class name_setter(type):

        def __new__(cls, _, bases, dict):
            return type.__new__(cls, name, bases, dict)

    class Stateful(link.Chain):

        __metaclass__ = name_setter

        def __init__(self, *args, **kwargs):
            super(Stateful, self).__init__(
                stateless=stateless_class(*args, **kwargs))
            self.state_names = self.stateless.state_names
            self.state_shapes = self.stateless.state_shapes
            self.state_name_to_idx = dict((name, i) for i, name
                                          in enumerate(self.state_names))
            self.reset_state()

        def to_cpu(self):
            for s in self.states:
                if s is not None:
                    s.to_cpu()

        def to_gpu(self, device=None):
            for s in self.states:
                if s is not None:
                    s.to_gpu(device)

        def set_state(self, *values):
            def _convert(x):
                assert isinstance(x, variable.Variable)
                x_ = x
                if self.xp is numpy:
                    x_.to_cpu()
                else:
                    x_.to_gpu()
                return x_

            assert len(values) == len(self.states)

            for i, value in enumerate(values):
                self.states[i] = _convert(value)

        def reset_state(self):
            self.states = [None] * len(self.state_names)

        def __getattr__(self, name):
            if name in self.state_names:
                idx = self.state_name_to_idx[name]
                return self.states[idx]
            elif hasattr(self.stateless, name):
                return getattr(self.stateless, name)
            else:
                return super(Stateful, self).__getattr__(name)

        def __call__(self, x):
            for i, val in enumerate(self.states):
                if self.states[i] is None:
                    self.states[i] = self.xp.zeros(
                        (len(x.data),) + self.state_shapes[i],
                        dtype=numpy.float32)

            args = tuple(self.states) + (x,)
            self.states = _force_tuple(self.stateless(*args))
            return self.states[-1]

    return Stateful
