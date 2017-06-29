import unittest


import mock
import numpy

import chainer
from chainer import functions
from chainer import links
from chainer import testing
from chainer.testing import attr


# testing.parameterize takes a list of dictionaries.
# Currently, we cannot set a function to the value of the dictionaries.
# As a workaround, we wrap the function and invoke it in __call__ method.
# See issue #1337 for detail.
class AccuracyWithIgnoreLabel(object):

    def __call__(self, y, t):
        return functions.accuracy(y, t, ignore_label=1)


@testing.parameterize(*testing.product({
    'accfun': [AccuracyWithIgnoreLabel(), None],
    'compute_accuracy': [True, False],
    'x_num': [1, 2],
    'label_key': [-1, 't'],
}))
class TestClassifier(unittest.TestCase):

    def setUp(self):
        if self.accfun is None:
            self.link = links.Classifier(
                chainer.Link(), label_key=self.label_key)
        else:
            self.link = links.Classifier(
                chainer.Link(), accfun=self.accfun, label_key=self.label_key)
        self.link.compute_accuracy = self.compute_accuracy

        self.x = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.t = numpy.random.randint(3, size=5).astype(numpy.int32)

    def check_call(self):
        xp = self.link.xp

        y = chainer.Variable(xp.random.uniform(
            -1, 1, (5, 7)).astype(numpy.float32))
        self.link.predictor = mock.MagicMock(return_value=y)

        x = chainer.Variable(xp.asarray(self.x))
        t = chainer.Variable(xp.asarray(self.t))
        if self.label_key == -1:
            if self.x_num == 1:
                loss = self.link(x, t)
                self.link.predictor.assert_called_with(x)
            elif self.x_num == 2:
                x_ = chainer.Variable(xp.asarray(self.x.copy()))
                loss = self.link(x, x_, t)
                self.link.predictor.assert_called_with(x, x_)
        elif self.label_key == 't':
            if self.x_num == 1:
                loss = self.link(x=x, t=t)
                self.link.predictor.assert_called_with(x=x)
            elif self.x_num == 2:
                x_ = chainer.Variable(xp.asarray(self.x.copy()))
                loss = self.link(x=x, y=x_, t=t)
                self.link.predictor.assert_called_with(x=x, y=x_)

        self.assertTrue(hasattr(self.link, 'y'))
        self.assertIsNotNone(self.link.y)

        self.assertTrue(hasattr(self.link, 'loss'))
        xp.testing.assert_allclose(self.link.loss.data, loss.data)

        self.assertTrue(hasattr(self.link, 'accuracy'))
        if self.compute_accuracy:
            self.assertIsNotNone(self.link.accuracy)
        else:
            self.assertIsNone(self.link.accuracy)

    def test_call_cpu(self):
        self.check_call()

    @attr.gpu
    def test_call_gpu(self):
        self.link.to_gpu()
        self.check_call()


class TestInvalidArgument(unittest.TestCase):

    def setUp(self):
        self.link = links.Classifier(links.Linear(10, 3))
        self.x = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)

    def check_invalid_argument(self):
        x = chainer.Variable(self.link.xp.asarray(self.x))
        with self.assertRaisesRegexp(TypeError, 'missing'):
            self.link(x)

    def test_invalid_argument_cpu(self):
        self.check_invalid_argument()

    @attr.gpu
    def test_invalid_argument_gpu(self):
        self.link.to_gpu()
        self.check_invalid_argument()


class TestInvalidLabelKey(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)

    def test_invalid_label_key_type(self):
        with self.assertRaisesRegexp(
                TypeError, 'label_key must be int or str'):
            links.Classifier(links.Linear(10, 3), label_key=None)

    def check_invalid_key(self, gpu, label_key):
        link = links.Classifier(links.Linear(10, 3), label_key=label_key)
        if gpu:
            link.to_gpu()
        x = chainer.Variable(link.xp.asarray(self.x))
        with self.assertRaisesRegexp(ValueError, 'Label key'):
            link(x)

    def test_invalid_index_cpu(self):
        self.check_invalid_key(False, 1)

    @attr.gpu
    def test_invalid_argument_gpu(self):
        self.check_invalid_key(True, 1)

    def test_invalid_index_too_small_cpu(self):
        self.check_invalid_key(False, -2)

    @attr.gpu
    def test_invalid_index_too_small_gpu(self):
        self.check_invalid_key(True, -2)

    def test_invalid_str_key_cpu(self):
        self.check_invalid_key(False, 't')

    @attr.gpu
    def test_invalid_str_key_gpu(self):
        self.check_invalid_key(True, 't')


testing.run_module(__name__, __file__)
