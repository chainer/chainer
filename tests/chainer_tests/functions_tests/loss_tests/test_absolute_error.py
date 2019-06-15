import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer import utils


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'shape': [(), (1,), (1, 1), (4,), (4, 3), (4, 3, 2)],
}))
@testing.inject_backend_tests(
    ['test_forward', 'test_backward', 'test_double_backward'],
    # CPU
    [{}]
    # GPU
    + testing.product({
        'use_cuda': [True],
    })
    # ChainerX
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0'],
    })
)
class TestAbsoluteError(testing.FunctionTestCase):

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 1e-3, 'rtol': 1e-3})
            self.check_backward_options.update({'atol': 5e-2, 'rtol': 5e-2})
            self.check_double_backward_options.update(
                {'atol': 3e-1, 'rtol': 3e-1})

    def generate_inputs(self):
        x0 = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        # Add sufficient margin to prevent computational error
        diff = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        diff[abs(diff) < 0.02] = 0.5
        x1 = numpy.asarray(x0 + diff)
        return (x0, x1)

    def forward_expected(self, inputs):
        x0, x1 = inputs
        return utils.force_array(numpy.abs(x0 - x1), self.dtype),

    def forward(self, inputs, device):
        x0, x1 = inputs
        return functions.absolute_error(x0, x1),


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'shape': [(), (1,), (1, 1), (4,), (4, 3), (4, 3, 2)],
}))
class TestNonDefaultGPU(unittest.TestCase):

    # This test is for https://github.com/chainer/chainer/issues/4669

    def setUp(self):
        self.x0 = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        # Add sufficient margin to prevent computational error
        diff = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        diff[abs(diff) < 0.02] = 0.5
        self.x1 = numpy.asarray(self.x0 + diff)
        self.gy = numpy.random.random(self.shape).astype(self.dtype)

    @attr.multi_gpu(2)
    def test_backward_non_default_gpu(self):
        x0 = chainer.Variable(cuda.to_gpu(self.x0, 1))
        x1 = chainer.Variable(cuda.to_gpu(self.x1, 1))
        gy = cuda.to_gpu(self.gy, 1)
        with cuda.get_device_from_id(0):
            y = functions.absolute_error(x0, x1)
            y.grad = gy
            y.backward()


testing.run_module(__name__, __file__)
