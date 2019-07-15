import unittest

import numpy

import chainer
from chainer.backends import _cpu
from chainer import functions as F
from chainer import link_hooks
from chainer import links as L
from chainer import testing
from chainer import variable

_inject_backend_tests = testing.inject_backend_tests(
    ['test_check_snnl', 'test_check_optimized_temp_snnl'],
    # CPU tests
    [
        {},
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ['never', 'always'],
    })
    # ChainerX tests
    + [
    ])


@_inject_backend_tests
class TestSNNLhook(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (10, 5)).astype(numpy.float32)
        self.t = numpy.random.randint(3, size=10).astype(numpy.int32)
        self.temperature = numpy.array(100).astype(numpy.float32)
        self.layer = L.Linear(5, 15)

    def _init_hook(self, optimize_temperature):
        self.hook = link_hooks.SNNL_hook(
            optimize_temperature=optimize_temperature)
        self.layer.add_hook(self.hook)

    def test_check_snnl(self, backend_config):
        self._init_hook(False)
        self.layer.to_device(backend_config.device)
        x = backend_config.get_array(self.x)
        t = backend_config.get_array(self.t)
        temperature = backend_config.get_array(self.temperature)
        self.hook.set_t(t)
        y = self.layer(x)

        snn_loss = F.soft_nearest_neighbor_loss(y, t, temperature, False)
        loss_hook = _cpu._to_cpu(self.hook.get_loss().data),
        loss_func = _cpu._to_cpu(snn_loss.data)
        testing.assert_allclose(loss_hook, loss_func)

    def test_check_optimized_temp_snnl(self, backend_config):
        self._init_hook(True)
        self.layer.to_device(backend_config.device)
        x = backend_config.get_array(self.x)
        t = backend_config.get_array(self.t)
        temperature = backend_config.get_array(self.temperature)
        self.hook.set_t(t)
        y = self.layer(x)

        _t = variable.Variable(
            backend_config.get_array(
                numpy.asarray([1], dtype=numpy.float32)))
        initial_temp = temperature

        def inverse_temp(_t):
            return initial_temp / _t

        ent_loss = F.soft_nearest_neighbor_loss(y, t, inverse_temp(_t), False)
        grad_t = chainer.grad([ent_loss], [_t])[0]
        if grad_t is not None:
            updated_t = _t - 0.1 * grad_t
        else:
            updated_t = _t

        inverse_t = inverse_temp(updated_t).data
        snn_loss = F.soft_nearest_neighbor_loss(y, t, inverse_t, False)
        loss_hook = _cpu._to_cpu(self.hook.get_loss().data)
        loss_func = _cpu._to_cpu(snn_loss.data)
        testing.assert_allclose(loss_hook, loss_func)


testing.run_module(__name__, __file__)
