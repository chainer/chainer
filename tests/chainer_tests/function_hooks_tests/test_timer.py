import os
import time
import unittest

import numpy
import six

import chainer
from chainer.backends import cuda
from chainer import function_hooks
from chainer import functions
from chainer.functions.math import basic_math
from chainer import testing
from chainer.testing import attr

try:
    _get_time = time.perf_counter
except AttributeError:
    if os.name == 'nt':
        _get_time = time.clock
    else:
        _get_time = time.time


def check_history(self, t, function_type, return_type):
    func_name = t[0]
    assert func_name == function_type.__name__
    assert isinstance(t[1], return_type)


class SimpleLink(chainer.Link):

    def __init__(self):
        super(SimpleLink, self).__init__()
        with self.init_scope():
            init_w = numpy.random.uniform(-1, 1, (3, 5)).astype(
                numpy.float32)
            self.w = chainer.Parameter(init_w)

    def forward(self, x):
        return self.w * x


class TestTimerHookToLink(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.TimerHook()
        self.layer = SimpleLink()
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)

    def test_name(self):
        assert self.h.name == 'TimerHook'

    def check_forward(self, x):
        with self.h:
            self.layer(chainer.Variable(x))
        assert len(self.h.call_history) == 1
        check_history(self, self.h.call_history[0], basic_math.Mul, float)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.layer.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x, gy):
        x = chainer.Variable(x)
        y = self.layer(x)
        y.grad = gy
        with self.h:
            y.backward()
        # It includes forward of + that accumulates gradients to W and b
        assert len(self.h.call_history) == 3
        for entry in self.h.call_history:
            if entry[0] == 'Add':
                continue
            check_history(self, entry, basic_math.Mul, float)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.layer.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


class TestTimerHookToFunction(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.TimerHook()
        self.f = functions.math.exponential.Exp()
        self.f.add_hook(self.h)
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)

    def check_forward(self, x):
        self.f.apply((chainer.Variable(x),))
        assert len(self.h.call_history) == 1
        check_history(self, self.h.call_history[0],
                      functions.math.exponential.Exp, float)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x, gy):
        x = chainer.Variable(x)
        y = self.f.apply((x,))[0]
        y.grad = gy
        y.backward()
        assert len(self.h.call_history) == 2
        check_history(self, self.h.call_history[1],
                      functions.math.exponential.Exp, float)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def test_reentrant(self):
        # In/grad data are random; these do not simulate the actually possible
        # cases.
        # any function other than Exp is ok
        g = functions.math.identity.Identity()

        self.h.backward_preprocess(self.f, (self.x,), (self.gy,))
        t1 = _get_time()
        time.sleep(0.001)  # longer than each hook call
        self.h.forward_preprocess(g, (self.x,))
        self.h.forward_postprocess(g, (self.x,))
        t2 = _get_time()
        self.h.backward_postprocess(self.f, (self.x,), (self.gy,))

        history = dict(self.h.call_history)
        assert len(history) == 2
        assert self.f._impl_name in history
        assert g._impl_name in history
        f_time = history[self.f._impl_name]
        g_time = history[g._impl_name]
        assert g_time <= t2 - t1
        assert f_time >= t2 - t1

    def test_reentrant_total_time(self):
        g = functions.math.identity.Identity()

        t0 = _get_time()
        self.h.backward_preprocess(self.f, (self.x,), (self.gy,))
        t1 = _get_time()
        self.h.forward_preprocess(g, (self.x,))
        time.sleep(0.001)
        self.h.forward_postprocess(g, (self.x,))
        t2 = _get_time()
        self.h.backward_postprocess(self.f, (self.x,), (self.gy,))
        t3 = _get_time()

        assert self.h.total_time() <= t3 - t0
        assert self.h.total_time() >= t2 - t1


class TestTimerPrintReport(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.TimerHook()
        self.f = functions.math.exponential.Exp()
        self.f.add_hook(self.h)
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)

    def test_summary(self):
        x = self.x
        self.f.apply((chainer.Variable(x),))
        self.f.apply((chainer.Variable(x),))
        assert len(self.h.call_history) == 2
        assert len(self.h.summary()) == 1

    def test_print_report(self):
        x = self.x
        self.f.apply((chainer.Variable(x),))
        self.f.apply((chainer.Variable(x),))
        io = six.StringIO()
        self.h.print_report(file=io)
        expect = r'''\AFunctionName  ElapsedTime  Occurrence
 +Exp +[0-9.\-e]+.s +[0-9]+$
'''
        actual = io.getvalue()
        six.assertRegex(self, actual, expect)


testing.run_module(__name__, __file__)
