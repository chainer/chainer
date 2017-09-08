import unittest

import mock
import numpy
import six

import chainer
from chainer import cuda
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
}))
class TestClassifier(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)
        self.t = numpy.random.randint(3, size=5).astype(numpy.int32)
        self.y = numpy.random.uniform(-1, 1, (5, 7)).astype(numpy.float32)

    def check_call(
            self, gpu, label_key, args, kwargs, model_args, model_kwargs):
        init_kwargs = {'label_key': label_key}
        if self.accfun is not None:
            init_kwargs['accfun'] = self.accfun
        link = links.Classifier(chainer.Link(), **init_kwargs)

        if gpu:
            xp = cuda.cupy
            link.to_gpu()
        else:
            xp = numpy

        link.compute_accuracy = self.compute_accuracy

        y = chainer.Variable(self.y)
        link.predictor = mock.MagicMock(return_value=y)

        loss = link(*args, **kwargs)
        link.predictor.assert_called_with(*model_args, **model_kwargs)

        self.assertTrue(hasattr(link, 'y'))
        self.assertIsNotNone(link.y)

        self.assertTrue(hasattr(link, 'loss'))
        xp.testing.assert_allclose(link.loss.data, loss.data)

        self.assertTrue(hasattr(link, 'accuracy'))
        if self.compute_accuracy:
            self.assertIsNotNone(link.accuracy)
        else:
            self.assertIsNone(link.accuracy)

    def test_call_cpu(self):
        self.check_call(
            False, -1, (self.x, self.t), {}, (self.x,), {})

    def test_call_three_args_cpu(self):
        self.check_call(
            False, -1, (self.x, self.x, self.t), {}, (self.x, self.x), {})

    def test_call_positive_cpu(self):
        self.check_call(
            False, 2, (self.x, self.x, self.t), {}, (self.x, self.x), {})

    def test_call_kwargs_cpu(self):
        self.check_call(
            False, 't', (self.x,), {'t': self.t}, (self.x,), {})

    def test_call_no_arg_cpu(self):
        self.check_call(
            False, 0, (self.t,), {}, (), {})

    @attr.gpu
    def test_call_gpu(self):
        self.to_gpu()
        self.check_call(
            True, -1, (self.x, self.t), {}, (self.x,), {})

    @attr.gpu
    def test_call_three_args_gpu(self):
        self.to_gpu()
        self.check_call(
            True, -1, (self.x, self.x, self.t), {}, (self.x, self.x), {})

    @attr.gpu
    def test_call_positive_gpu(self):
        self.to_gpu()
        self.check_call(
            True, 2, (self.x, self.x, self.t), {}, (self.x, self.x), {})

    @attr.gpu
    def test_call_kwargs_gpu(self):
        self.to_gpu()
        self.check_call(
            True, 't', (self.x,), {'t': self.t}, (self.x,), {})

    @attr.gpu
    def test_call_no_arg_gpu(self):
        self.to_gpu()
        self.check_call(
            True, 0, (self.t,), {}, (), {})

    def to_gpu(self):
        self.x = cuda.to_gpu(self.x)
        self.t = cuda.to_gpu(self.t)
        self.y = cuda.to_gpu(self.y)


class TestInvalidArgument(unittest.TestCase):

    def setUp(self):
        self.link = links.Classifier(links.Linear(10, 3))
        self.x = numpy.random.uniform(-1, 1, (5, 10)).astype(numpy.float32)

    def check_invalid_argument(self):
        x = chainer.Variable(self.link.xp.asarray(self.x))
        with self.assertRaises(TypeError):
            # link.__call__ raises TypeError as the number of arguments
            # is illegal
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
        with six.assertRaisesRegex(
                self, TypeError, 'label_key must be int or str'):
            links.Classifier(links.Linear(10, 3), label_key=None)

    def check_invalid_key(self, gpu, label_key):
        link = links.Classifier(links.Linear(10, 3), label_key=label_key)
        if gpu:
            link.to_gpu()
        x = chainer.Variable(link.xp.asarray(self.x))
        with six.assertRaisesRegex(self, ValueError, 'Label key'):
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
