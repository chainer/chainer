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
# testing.parameterize takes a list of dictionaries.
# Currently, we cannot set a function to the value of the dictionaries.
# As a workaround, we wrap the function and invoke it in __call__ method.
# See issue #1337 for detail.
#class AccuracyWithIgnoreLabel(object):

#    def __call__(self, y, t):
#        return functions.accuracy(y, t, ignore_label=1)


@testing.parameterize(*testing.product({
    'link_names': [['0'], ['0','1'], None],
#    'compute_accuracy': [True, False],
}))
class TestClassifierSNNL(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (10, 5)).astype(numpy.float32)
        self.t = numpy.random.randint(3, size=10).astype(numpy.int32)

    def check_call(self, gpu):
#        init_kwargs = {'label_key': label_key}
#        if self.accfun is not None:
#            init_kwargs['accfun'] = self.accfun
        predictor = Sequential(L.Linear(5, 15), F.relu, L.Linear(15,10), F.relu, L.Linear(10,3))
        link = L.ClassifierSNNL(predictor, link_names=self.link_names)
#        link = L.ClassifierSNNL(predictor)

        if gpu:
            xp = cuda.cupy
            link.to_gpu()
        else:
            xp = numpy


        snn_loss = link(self.x, self.t)
        hook_names = []
        loss = link.loss
        snn_loss2 = loss
        print("hogehoge")
        for l in list(link.predictor.children()):
            print(l.name)
            for name, linkhook in l.local_link_hooks.items():
                hook_names.append(l.name)
                print(l.name)
                print(name)
                print(linkhook.get_loss())
                snn_loss2 += link.factor * linkhook.get_loss()
        print("hogehoge100")
        print(snn_loss.data)
        print(snn_loss2.data)
        print(link.snn_loss.data)

        print(hook_names)

        self.assertTrue(hasattr(link, 'y'))
        self.assertIsNotNone(link.y)

        self.assertTrue(hasattr(link, 'loss'))
        xp.testing.assert_allclose(link.snn_loss.data, snn_loss.data)
        xp.testing.assert_allclose(link.snn_loss.data, snn_loss2.data)

        if self.link_names is not None:
             self.assertEqual(self.link_names, hook_names)
        else:
             self.assertEqual([l.name for l in list(link.predictor.children())[:-1]] , hook_names)

#        self.assertTrue(hasattr(link, 'accuracy'))
#        if self.compute_accuracy:
#            self.assertIsNotNone(link.accuracy)
#        else:
#            self.assertIsNone(link.accuracy)


    def test_call_cpu(self):
        self.check_call(
            False)


testing.run_module(__name__, __file__)
