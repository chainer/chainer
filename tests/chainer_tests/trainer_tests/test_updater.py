import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import optimizers
from chainer import testing
from chainer.testing import attr
from chainer import trainer


class SimpleChain(chainer.Chain):

    def __init__(self):
        super(SimpleChain, self).__init__()
        self.add_param('w', 2)
        self.w.data[...] = [2, -3]

    def __call__(self, x, y):
        self.h = self.w * x + y  # vector
        self.z = functions.sum(self.h)  # scalar
        self.zz = self.z + 1  # scalar
        return self.zz


@testing.parameterize({'lossfun': True}, {'lossfun': False})
class TestStandardUpdater(unittest.TestCase):

    def setUp(self):
        self.target = SimpleChain()
        self.optimizer = optimizers.SGD(lr=1)
        self.optimizer.setup(self.target)
        self.x = numpy.array([20, -10], dtype='f')
        self.y = numpy.array([100, -50], dtype='f')
        if self.lossfun:
            def objective(x, y):
                return self.target(x, y)
            self.updater = trainer.StandardUpdater(objective)
        else:
            self.updater = trainer.StandardUpdater()

    def check_update(self, x, y):
        xp = cuda.get_array_module(x, y)
        result = self.updater((x, y), self.optimizer)
        xp.testing.assert_array_equal(self.target.w.data,
                                      xp.array([2 - 20, -3 + 10]))
        self.assertIsInstance(result, dict)
        self.assertNotIn('h', result)
        self.assertIn('z', result)
        self.assertIsInstance(result['z'], xp.ndarray)
        self.assertIn('zz', result)
        self.assertIsInstance(result['zz'], xp.ndarray)

    def test_update_cpu(self):
        self.check_update(self.x, self.y)

    @attr.gpu
    def test_update_gpu(self):
        x = cuda.to_gpu(self.x)
        y = cuda.to_gpu(self.y)
        self.target.to_gpu()
        self.check_update(x, y)
