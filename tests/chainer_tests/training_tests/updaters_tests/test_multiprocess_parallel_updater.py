import os
import subprocess
import sys
import unittest

import numpy

import chainer
from chainer.backends import cuda
import chainer.functions.math.minmax
from chainer import initializers
import chainer.reporter
from chainer import testing
from chainer.testing import attr
import chainer.training.updaters.multiprocess_parallel_updater as mpu

import copy


class SimpleNet(chainer.Chain):
    insize = 5

    def __init__(self, dtype=numpy.float32):
        super(SimpleNet, self).__init__()
        self.dtype = dtype
        W = initializers.HeNormal(1 / numpy.sqrt(2), self.dtype)
        bias = initializers.Zero(self.dtype)
        with self.init_scope():
            self.conv = chainer.links.Convolution2D(2, 2, 3, initialW=W,
                                                    initial_bias=bias)
            self.fc = chainer.links.Linear(18, 2, initialW=W,
                                           initial_bias=bias)
        self.train = True

    def clear(self):
        self.loss = None
        self.accuracy = None

    def __call__(self, x, t):
        h = chainer.functions.relu(self.conv(x))
        y = self.fc(h)

        self.loss = chainer.functions.softmax_cross_entropy(y, t)
        self.accuracy = chainer.functions.accuracy(y, t)

        return self.loss


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.float16],
}))
class TestGatherScatter(unittest.TestCase):

    def setUp(self):
        pass

    @attr.gpu
    def test_gather_scatter_grads(self):
        cupy = cuda.cupy
        model0 = SimpleNet(dtype=self.dtype)
        model1 = copy.deepcopy(model0)

        model0.to_gpu()
        model1.to_gpu()

        optimizer0 = chainer.optimizers.SGD(lr=1.0)
        optimizer0.setup(model0)

        optimizer1 = chainer.optimizers.SGD(lr=1.0)
        optimizer1.setup(model1)

        bsize = 8

        x = numpy.random.uniform(0, 1, (bsize, 2, 5, 5)).astype(self.dtype)
        t = numpy.empty(bsize, dtype=numpy.int32)
        for i in range(bsize):
            t[i] = i % 2

        x = chainer.Variable(chainer.backends.cuda.to_gpu(x))
        t = chainer.Variable(chainer.backends.cuda.to_gpu(t))

        loss0 = model0(x, t)

        model0.cleargrads()
        model1.cleargrads()

        loss0.backward()
        gg0 = mpu.gather_grads(model0)
        mpu.scatter_grads(model1, gg0)

        cupy.testing.assert_array_equal(model0.conv.W.grad, model1.conv.W.grad)
        cupy.testing.assert_array_equal(model0.conv.b.grad, model1.conv.b.grad)
        cupy.testing.assert_array_equal(model0.fc.W.grad, model1.fc.W.grad)
        cupy.testing.assert_array_equal(model0.fc.b.grad, model1.fc.b.grad)

        optimizer0.update()
        optimizer1.update()

        cupy.testing.assert_array_equal(model0.conv.W.data, model1.conv.W.data)
        cupy.testing.assert_array_equal(model0.conv.b.data, model1.conv.b.data)
        cupy.testing.assert_array_equal(model0.fc.W.data, model1.fc.W.data)
        cupy.testing.assert_array_equal(model0.fc.b.data, model1.fc.b.data)

    def test_gather_grads_raise_on_cpu(self):
        model = SimpleNet(dtype=self.dtype)
        with self.assertRaises(RuntimeError):
            mpu.gather_grads(model)

    @attr.gpu
    def test_gather_scatter_params(self):
        cupy = cuda.cupy
        model0 = SimpleNet(dtype=self.dtype)
        model1 = SimpleNet(dtype=self.dtype)

        model0.to_gpu()
        model1.to_gpu()

        gp0 = mpu.gather_params(model0)
        mpu.scatter_params(model1, gp0)

        cupy.testing.assert_array_equal(model0.conv.W.data, model1.conv.W.data)
        cupy.testing.assert_array_equal(model0.conv.b.data, model1.conv.b.data)
        cupy.testing.assert_array_equal(model0.fc.W.data, model1.fc.W.data)
        cupy.testing.assert_array_equal(model0.fc.b.data, model1.fc.b.data)

    def test_gather_params_raise_on_cpu(self):
        model = SimpleNet(dtype=self.dtype)
        with self.assertRaises(RuntimeError):
            mpu.gather_params(model)


def _run_test_snippet(name, *args):
    script_path = os.path.join(
        os.path.dirname(__file__), 'snippets/{}'.format(name))
    proc = subprocess.Popen(
        (sys.executable, script_path) + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    stdoutdata, stderrdata = proc.communicate()
    ret = proc.returncode
    return (ret, stdoutdata, stderrdata)


class TestRawArray(unittest.TestCase):

    def setUp(self):
        pass

    @attr.gpu
    @unittest.skipUnless(mpu.MultiprocessParallelUpdater.available(),
                         'MultiprocessParallelUpdater is not available.')
    def test_update_uses_raw_array(self):
        ret, stdoutdata, stderrdata = _run_test_snippet('raw_array.py')
        assert ret == 0, (
            '[stdout]:{!r}\n'
            '[stderr]:{!r}'.format(stdoutdata, stderrdata))


class TestChildReporter(unittest.TestCase):

    def check_with_gpus(self, n_devices):
        device_ids_str = ','.join([str(n) for n in range(n_devices)])
        ret, stdoutdata, stderrdata = _run_test_snippet(
            'child_reporter.py', device_ids_str)
        assert ret == 0, (
            '[stdout]:{!r}\n'
            '[stderr]:{!r}'.format(stdoutdata, stderrdata))

    @attr.gpu
    @unittest.skipUnless(mpu.MultiprocessParallelUpdater.available(),
                         'MultiprocessParallelUpdater is not available.')
    def test_single_device(self):
        self.check_with_gpus(1)

    @attr.multi_gpu(2)
    @unittest.skipUnless(mpu.MultiprocessParallelUpdater.available(),
                         'MultiprocessParallelUpdater is not available.')
    def test_multi_device(self):
        self.check_with_gpus(2)


class TestCUDAContext(unittest.TestCase):

    @attr.gpu
    @unittest.skipUnless(mpu.MultiprocessParallelUpdater.available(),
                         'MultiprocessParallelUpdater is not available.')
    def test_cuda_init(self):
        ret, stdoutdata, stderrdata = _run_test_snippet('cuda_init.py')
        assert ret == 0, (
            '[stdout]:{!r}\n'
            '[stderr]:{!r}'.format(stdoutdata, stderrdata))


testing.run_module(__name__, __file__)
