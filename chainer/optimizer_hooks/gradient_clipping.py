import collections

import numpy
import six

import chainer
from chainer import backend


def _sum_sqnorm_grads(params):
    # Calculates sum of squares of gradients.

    # Returns a tuple of the sum and the device of the sum.
    # The device will be `None` in multi-device case.

    # If the inputs are on a single device, the sum is returned
    # as an ndarray on the device, so that no synchronization is taken place.
    # If there are multiple devices, accumulation is done on each device
    # first, and the total sum is returned as a python float.

    # TODO(niboshi): Support and test len(params) == 0

    params_grouped = collections.defaultdict(list)
    devices_map = {}

    # Group params by devices.
    for param in params:
        device = param.device
        params_grouped[device.name].append(param)
        devices_map[device.name] = device

    # Calculates partial sums for each device.
    sq_sums = []
    for device_name, paramlist in six.iteritems(params_grouped):
        device = devices_map[device_name]
        with chainer.using_device(device):
            dots = []
            for param in paramlist:
                g = param.grad
                g = g.ravel()
                dots.append(g.dot(g))
            sq_sums.append(sum(dots))

    # Return the total sum.
    if len(sq_sums) == 1:
        # single device
        sqnorm = sq_sums[0]
        ret_device = params[0].device
    else:
        # multi-device
        sqnorm = sum([float(s) for s in sq_sums])
        ret_device = None
    return sqnorm, ret_device


class GradientClipping(object):
    """Optimizer hook function for gradient clipping.

    This hook function scales all gradient arrays to fit to the defined L2 norm
    threshold.

    Args:
        threshold (float): L2 norm threshold.

    Attributes:
        ~optimizer_hooks.GradientClipping.threshold (float): L2
                         norm threshold of gradient norm.
        ~optimizer_hooks.GradientClipping.timing (string): Specifies
                         when this hook should be
                         called by the Optimizer/UpdateRule. Valid values are
                         'pre' (before any updates) and 'post' (after any
                         updates).

    .. versionadded:: 4.0.0
       The *timing* parameter.

    """
    name = 'GradientClipping'
    timing = 'pre'

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, opt):
        sqnorm, device = _sum_sqnorm_grads(list(opt.target.params(False)))
        if device is None:
            # Assign a dummy device for using_device.
            device = backend.CpuDevice()

        with chainer.using_device(device):
            norm = device.xp.sqrt(sqnorm)
            # TODO(niboshi): Could be inf if norm == 0
            rate = self.threshold / norm
            # In NumPy backend, `rate` is already available on CPU and thus
            # can be compared against 1 without extra overhead.
            # Otherwise `clip` is used to avoid synchronization.
            if device.xp is numpy:
                if rate >= 1:
                    return
            else:
                rate = rate.clip(None, 1)

        for param in opt.target.params(False):
            grad = param.grad
            with chainer.using_device(param.device):
                grad *= rate
