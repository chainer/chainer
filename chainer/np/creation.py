import chainer
from chainer import np


def empty(shape, dtype=float, order='C', device=None):
    with np.get_device(device) as dev:
        return chainer.as_variable(dev.xp.empty(shape, dtype, order))


def empty_like(a, dtype=None, device=None):
    with np.get_device(device) as dev:
        return chainer.as_variable(dev.xp.empty_like(a, dtype))


def eye(N, M=None, k=0, dtype=float, device=None):
    with np.get_device(device) as dev:
        return chainer.as_variable(dev.xp.eye(N, M, k, dtype))


def identity(n, dtype=float, device=None):
    with np.get_device(device) as dev:
        return chainer.as_variable(dev.xp.identity(n, dtype))


def ones(shape, dtype=float, device=None):
    with np.get_device(device) as dev:
        return chainer.as_variable(dev.xp.ones(shape, dtype))


def ones_like(a, dtype=float, device=None):
    with np.get_device(device) as dev:
        return chainer.as_variable(dev.xp.ones_like(a, dtype))


def zeros(shape, dtype=float, device=None):
    with np.get_device(device) as dev:
        return chainer.as_variable(dev.xp.zeros(shape, dtype))


def zeros_like(a, dtype=float, device=None):
    with np.get_device(device) as dev:
        return chainer.as_variable(dev.xp.zeros_like(a, dtype))


def full(shape, fill_value, dtype=float, device=None):
    with np.get_device(device) as dev:
        return chainer.as_variable(dev.xp.full(shape, fill_value, dtype))


def full_like(a, fill_value, dtype=float, device=None):
    with np.get_device(device) as dev:
        return chainer.as_variable(dev.xp.full_like(a, fill_value, dtype))
