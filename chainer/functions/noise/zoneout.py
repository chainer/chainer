import numpy

from chainer import backend
from chainer import configuration
from chainer import function_node
from chainer.utils import argument
from chainer.utils import type_check


class Zoneout(function_node.FunctionNode):

    """Zoneout regularization."""

    def __init__(self, zoneout_ratio):
        self.zoneout_ratio = zoneout_ratio

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

    def forward(self, inputs):
        self.retain_inputs(())

        h, x = inputs
        xp = backend.get_array_module(*x)
        if xp is numpy:
            flag_x = xp.random.rand(*x.shape) >= self.zoneout_ratio
        else:
            flag_x = (xp.random.rand(*x.shape) >=
                      self.zoneout_ratio)
        self.flag_h = xp.ones_like(flag_x) ^ flag_x
        self.flag_x = flag_x
        return h * self.flag_h + x * self.flag_x,

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        ret = []
        if 0 in indexes:
            ret.append(gy * self.flag_h)
        if 1 in indexes:
            ret.append(gy * self.flag_x)
        return ret


def zoneout(h, x, ratio=.5, **kwargs):
    """zoneout(h, x, ratio=.5)

    Drops elements of input variable and sets to previous variable randomly.

    This function drops input elements randomly with probability ``ratio`` and
    instead sets dropping element to their previous variable. In testing mode ,
    it does nothing and just returns ``x``.

    Args:
        h (:class:`~chainer.Variable` or :ref:`ndarray`): Previous variable.
        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.
        ratio (float): Zoneout ratio.

    Returns:
        ~chainer.Variable: Output variable.

    See the paper: `Zoneout: Regularizing RNNs by Randomly Preserving Hidden
    Activations <https://arxiv.org/abs/1606.01305>`_.

    """
    if kwargs:
        argument.check_unexpected_kwargs(
            kwargs, train='train argument is not supported anymore. '
            'Use chainer.using_config')
        argument.assert_kwargs_empty(kwargs)

    if configuration.config.train:
        return Zoneout(ratio).apply((h, x))[0]
    return x
