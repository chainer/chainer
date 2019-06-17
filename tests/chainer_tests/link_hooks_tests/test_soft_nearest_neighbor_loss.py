import unittest
import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import links as L
from chainer import functions as F
from chainer import link_hooks
from chainer import testing
from chainer.testing import attr
from chainer import variable

class TestSNNLhook(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (10, 5)).astype(numpy.float32)
        self.t = numpy.random.randint(3, size=10).astype(numpy.int32)
        self.temperature = 100.
        self.layer = L.Linear(5, 15)

    def _init_hook(self,optimize_temperature):
        self.hook = link_hooks.SNNL_hook(optimize_temperature=optimize_temperature)
        self.layer.add_hook(self.hook)

    def test_check_snnl(self):
        xp = numpy
        self._init_hook(False)
        self.hook.set_t(self.t)
        y = self.layer(self.x)
        snn_loss = F.soft_nearest_neighbor_loss(y , self.t, self.temperature, False)
        xp.testing.assert_allclose(self.hook.get_loss().data, snn_loss.data)

    def test_check_optimized_temp_snnl(self):
        xp = numpy
        self._init_hook(True)
        self.hook.set_t(self.t)
        y = self.layer(self.x)

        t = variable.Variable(xp.asarray([1], dtype=xp.float32))
        initial_temp = self.temperature

        def inverse_temp(t):
            # pylint: disable=missing-docstring
            # we use inverse_temp because it was observed to be more stable
            # when optimizing.
            return initial_temp / t

        ent_loss = F.soft_nearest_neighbor_loss(y, self.t, inverse_temp(t), False)
        grad_t = chainer.grad([ent_loss], [t])[0]
        if grad_t is not None:
            updated_t = t - 0.1 * grad_t
        else:
            updated_t = t

        inverse_t = inverse_temp(updated_t).data
        snn_loss = F.soft_nearest_neighbor_loss(y, self.t, inverse_t, False)
        xp.testing.assert_allclose(self.hook.get_loss().data, snn_loss.data)


testing.run_module(__name__, __file__)
