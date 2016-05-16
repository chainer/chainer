import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import links
from chainer import testing
from chainer.testing import attr


@testing.parameterize(
    {'in_size': 10, 'n_layers': 2},
    {'in_size': 10, 'n_layers': 3},
)
class TestNStepLSTM(unittest.TestCase):

    def setUp(self):
        self.link = links.NStepLSTM(self.in_size, self.n_layers)
        self.link.zerograds()

        x_shape = (1, 4, self.in_size)
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)

        self.w = self.link.w
        self.b = self.link.b

    def check_forward(self, x_data):
        xp = self.link.xp
        x = chainer.Variable(x_data)

        y1 = self.link(x)
        c0 = chainer.variable.Variable(
            xp.zeros(
                (self.n_layers, x.data.shape[1], self.in_size),
                dtype=x.data.dtype))
        h0 = chainer.variable.Variable(
            xp.zeros(
                (self.n_layers, x.data.shape[1], self.in_size),
                dtype=x.data.dtype))

        h1_expect, c1_expect, y1_expect = functions.n_step_lstm(
            self.n_layers, h0, c0, x, self.w, self.b)

        gradient_check.assert_allclose(y1.data, y1_expect.data)
        gradient_check.assert_allclose(self.link.h.data, h1_expect.data)
        gradient_check.assert_allclose(self.link.c.data, c1_expect.data)

        h1 = self.link.h
        c1 = self.link.c

        y2 = self.link(x)

        h2_expect, c2_expect, y2_expect = functions.n_step_lstm(
            self.n_layers, h1, c1, x, self.w, self.b)
        gradient_check.assert_allclose(y2.data, y2_expect.data)
        gradient_check.assert_allclose(self.link.h.data, h2_expect.data)
        gradient_check.assert_allclose(self.link.c.data, c2_expect.data)

    @attr.gpu
    def test_forward_gpu(self):
        self.link.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))


class TestNStepLSSTMRestState(unittest.TestCase):

    def setUp(self):
        self.link = links.NStepLSTM(5, 2)
        self.x = chainer.Variable(
            numpy.random.uniform(-1, 1, (1, 3, 5)).astype(numpy.float32))

    def check_state(self):
        self.assertIsNone(self.link.c)
        self.assertIsNone(self.link.h)
        self.link(self.x)
        self.assertIsNotNone(self.link.c)
        self.assertIsNotNone(self.link.h)

    @attr.gpu
    def test_state_gpu(self):
        self.link.to_gpu()
        self.x.to_gpu()
        self.check_state()

    @attr.gpu
    def check_reset_state(self):
        self.link.to_gpu()
        self.x.to_gpu()
        self.link(self.x)
        self.link.reset_state()
        self.assertIsNone(self.link.c)
        self.assertIsNone(self.link.h)

    @attr.gpu
    def test_reset_state_gpu(self):
        self.link.to_gpu()
        self.x.to_gpu()
        self.check_reset_state()


class TestNStepLSTMToCPUToGPU(unittest.TestCase):

    def setUp(self):
        self.link = links.NStepLSTM(5, 7)
        self.x = chainer.Variable(
            numpy.random.uniform(-1, 1, (1, 3, 5)).astype(numpy.float32))

    def check_to_cpu(self, s):
        self.link.to_cpu()
        self.assertIsInstance(s.data, self.link.xp.ndarray)
        self.link.to_cpu()
        self.assertIsInstance(s.data, self.link.xp.ndarray)

    @attr.gpu
    def test_to_cpu_cpu(self):
        self.link.to_gpu()
        self.x.to_gpu()
        self.link(self.x)

        # bring values to cpu.
        self.link.to_cpu()
        self.check_to_cpu(self.link.c)
        self.check_to_cpu(self.link.h)

        # call to_cpu from cpu.
        self.link.to_cpu()
        self.check_to_cpu(self.link.c)
        self.check_to_cpu(self.link.h)

    @attr.gpu
    def test_to_cpu_gpu(self):
        self.link.to_gpu()
        self.x.to_gpu()
        self.link(self.x)
        self.check_to_cpu(self.link.c)
        self.check_to_cpu(self.link.h)

    def check_to_cpu_to_gpu(self, s):
        self.link.to_gpu()
        self.assertIsInstance(s.data, self.link.xp.ndarray)
        self.link.to_gpu()
        self.assertIsInstance(s.data, self.link.xp.ndarray)
        self.link.to_cpu()
        self.assertIsInstance(s.data, self.link.xp.ndarray)
        self.link.to_gpu()
        self.assertIsInstance(s.data, self.link.xp.ndarray)

    @attr.gpu
    def test_to_cpu_to_gpu_gpu(self):
        self.link.to_gpu()
        self.x.to_gpu()
        self.link(self.x)
        self.check_to_cpu_to_gpu(self.link.c)
        self.check_to_cpu_to_gpu(self.link.h)


testing.run_module(__name__, __file__)
