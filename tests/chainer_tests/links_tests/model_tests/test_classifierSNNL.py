import unittest

import mock
import numpy
import six
  
import chainer
from chainer.backends import cuda
from chainer import functions as F
from chainer import links as L
from chainer import testing
from chainer.testing import attr

from chainer import Sequential

@testing.parameterize(*testing.product({
    'link_names': [['0'], ['0','1'], None],
}))

class TestClassifierSNNL(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (10, 5)).astype(numpy.float32)
        self.t = numpy.random.randint(3, size=10).astype(numpy.int32)

    def check_call(self, gpu):
        predictor = Sequential(L.Linear(5, 15), F.relu, L.Linear(15,10), F.relu, L.Linear(10,3))
        link = L.ClassifierSNNL(predictor, link_names=self.link_names)

        if gpu:
            xp = cuda.cupy
            link.to_gpu()
        else:
            xp = numpy


        snn_loss = link(self.x, self.t)
        hook_names = []
        loss = link.loss
        snn_loss_expected = loss
        for l in list(link.predictor.children()):
            for name, linkhook in l.local_link_hooks.items():
                hook_names.append(l.name)
                snn_loss_expected += link.factor * linkhook.get_loss()

        self.assertTrue(hasattr(link, 'y'))
        self.assertIsNotNone(link.y)

        self.assertTrue(hasattr(link, 'loss'))
        xp.testing.assert_allclose(link.snn_loss.data, snn_loss_expected.data)

        if self.link_names is not None:
             self.assertEqual(self.link_names, hook_names)
        else:
             self.assertEqual([l.name for l in list(link.predictor.children())[:-1]] , hook_names)

    def test_call_cpu(self):
        self.check_call(
            False)

    @attr.gpu
    def test_call_gpu(self):
        self.to_gpu()
        self.check_call(
            True)

    def to_gpu(self):
        self.x = cuda.to_gpu(self.x)
        self.t = cuda.to_gpu(self.t)

testing.run_module(__name__, __file__)
