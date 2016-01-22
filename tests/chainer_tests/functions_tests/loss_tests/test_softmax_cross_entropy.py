import math
import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition


class TestSoftmaxCrossEntropy(unittest.TestCase):

    backward_atol = 1e-4

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (4, 3)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 3, (4,)).astype(numpy.int32)

    def check_forward(self, x_data, t_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        loss = functions.softmax_cross_entropy(x, t, use_cudnn)
        self.assertEqual(loss.data.shape, ())
        self.assertEqual(loss.data.dtype, numpy.float32)
        loss_value = float(cuda.to_cpu(loss.data))

        # Compute expected value
        loss_expect = 0.0
        count = 0
        for i in six.moves.range(self.x.shape[0]):
            if self.t[i] == -1:
                continue
            log_z = numpy.ufunc.reduce(numpy.logaddexp, self.x[i])
            loss_expect -= (self.x[i] - log_z)[self.t[i]]
            count += 1

        if count == 0:
            loss_expect = 0.0
        else:
            loss_expect /= count

        self.assertAlmostEqual(loss_expect, loss_value, places=5)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x, self.t)

    @attr.cudnn
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu_no_cudnn(self):
        self.check_forward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), False)

    def check_backward(self, x_data, t_data, use_cudnn=True):
        gradient_check.check_backward(
            functions.SoftmaxCrossEntropy(use_cudnn),
            (x_data, t_data), None, eps=0.02, atol=self.backward_atol)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.t)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t))

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_no_cudnn(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.t), False)


class TestSoftmaxCrossEntropyUnstable(TestSoftmaxCrossEntropy):

    backward_atol = 1e-3

    def setUp(self):
        self.x = numpy.array([[-1000, 1]], dtype=numpy.float32)
        self.t = numpy.array([0], dtype=numpy.int32)


class TestReplicatedSoftmaxCrossEntropy1(TestSoftmaxCrossEntropy):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (4, 3, 2)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 3, (4, 2)).astype(numpy.int32)

    def check_forward(self, x_data, t_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        loss = functions.softmax_cross_entropy(
            x, t, use_cudnn, normalize=True)
        self.assertEqual(loss.data.shape, ())
        self.assertEqual(loss.data.dtype, numpy.float32)
        loss_value = float(cuda.to_cpu(loss.data))

        # Compute expected value
        y = numpy.exp(self.x)
        loss_expect = 0.0
        count = 0
        for i in six.moves.range(y.shape[0]):
            for k in six.moves.range(y.shape[2]):
                if self.t[i, k] == -1:
                    continue
                loss_expect -= math.log(
                    y[i, self.t[i, k], k] / y[i, :, k].sum())
                count += 1

        if count == 0:
            loss_expect = 0.0
        else:
            loss_expect /= count

        self.assertAlmostEqual(loss_expect, loss_value, places=4)


class TestReplicatedSoftmaxCrossEntropy2(TestSoftmaxCrossEntropy):

    def setUp(self):
        self.x = numpy.random.uniform(
            -1, 1, (4, 3, 2, 5)).astype(numpy.float32)
        self.t = numpy.random.randint(0, 3, (4, 2, 5)).astype(numpy.int32)

    def check_forward(self, x_data, t_data, use_cudnn=True):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        loss = functions.softmax_cross_entropy(
            x, t, use_cudnn, normalize=False)
        self.assertEqual(loss.data.shape, ())
        self.assertEqual(loss.data.dtype, numpy.float32)
        loss_value = float(cuda.to_cpu(loss.data))

        # Compute expected value
        y = numpy.exp(self.x)
        loss_expect = 0.0
        for i in six.moves.range(y.shape[0]):
            for k in six.moves.range(y.shape[2]):
                for l in six.moves.range(y.shape[3]):
                    if self.t[i, k, l] == -1:
                        continue
                    loss_expect -= math.log(
                        y[i, self.t[i, k, l], k, l] / y[i, :, k, l].sum())
        loss_expect /= y.shape[0]

        self.assertAlmostEqual(loss_expect, loss_value, places=4)


class TestSoftmaxCrossEntropyWithIgnoreLabel(TestSoftmaxCrossEntropy):

    def setUp(self):
        super(TestSoftmaxCrossEntropyWithIgnoreLabel, self).setUp()
        self.t[2] = -1


class TestSoftmaxCrossEntropyIgnoreAll(TestSoftmaxCrossEntropy):

    def setUp(self):
        super(TestSoftmaxCrossEntropyIgnoreAll, self).setUp()
        self.t[:] = -1


class TestReplicatedSoftmaxCrossEntropy1IgnoreLabel(
        TestReplicatedSoftmaxCrossEntropy1):

    def setUp(self):
        super(TestReplicatedSoftmaxCrossEntropy1IgnoreLabel, self).setUp()
        self.t[0, 1] = -1


class TestReplicatedSoftmaxCrossEntropy2IgnoreLabel(
        TestReplicatedSoftmaxCrossEntropy2):

    def setUp(self):
        super(TestReplicatedSoftmaxCrossEntropy2IgnoreLabel, self).setUp()
        self.t[0, 1, 2] = -1


class TestReplicatedSoftmaxCrossEntropy1IgnoreAll(
        TestReplicatedSoftmaxCrossEntropy1):

    def setUp(self):
        super(TestReplicatedSoftmaxCrossEntropy1IgnoreAll, self).setUp()
        self.t[:] = -1


class TestReplicatedSoftmaxCrossEntropy2IgnoreAll(
        TestReplicatedSoftmaxCrossEntropy2):

    def setUp(self):
        super(TestReplicatedSoftmaxCrossEntropy2IgnoreAll, self).setUp()
        self.t[:] = -1


testing.run_module(__name__, __file__)
