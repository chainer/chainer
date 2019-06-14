import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer import utils


@testing.parameterize(*testing.product({
    'shape': [(3, 2), ()],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'contiguous': [None, 'C'],
}))
@testing.fix_random()
@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
        {'use_ideep': 'always'}
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'use_cudnn': ('never', 'always'),
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1']
    })
)
class TestTanh(testing.FunctionTestCase):

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_backward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_double_backward_options = {'atol': 5e-3, 'rtol': 5e-2}

    def generate_inputs(self):
        x = numpy.random.uniform(-.5, .5, self.shape).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        return functions.tanh(x),

    def forward_expected(self, inputs):
        x, = inputs
        return utils.force_array(numpy.tanh(x)),


@testing.parameterize(*testing.product({
    'use_cudnn': ['always', 'auto', 'never'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestTanhCudnnCall(unittest.TestCase):

    def setUp(self):
        self.x = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        self.gy = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.expect = chainer.should_use_cudnn('==always')

    def forward(self):
        x = chainer.Variable(self.x)
        return functions.tanh(x)

    def test_call_cudnn_forward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            default_func = cuda.cupy.cudnn.activation_forward
            with testing.patch('cupy.cudnn.activation_forward') as func:
                func.side_effect = default_func
                self.forward()
                self.assertEqual(func.called, self.expect)

    def test_call_cudnn_backward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            y = self.forward()
            y.grad = self.gy
            default_func = cuda.cupy.cudnn.activation_backward
            with testing.patch('cupy.cudnn.activation_backward') as func:
                func.side_effect = default_func
                y.backward()
                self.assertEqual(func.called, self.expect)


testing.run_module(__name__, __file__)
