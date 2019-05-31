import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product_dict(
    [
        {'shape': None, 'axis': 1},
        {'shape': (5,), 'axis': 0},
        {'shape': (2, 3), 'axis': 0},
        {'shape': (2, 3), 'axis': 1},
        {'shape': (2, 3, 4), 'axis': 0},
        {'shape': (2, 3, 4), 'axis': -1},
        {'shape': (2, 3, 2, 3), 'axis': -3},
        {'shape': (2, 3, 2, 3), 'axis': 3},
    ],
    testing.product({
        'dtype': [numpy.float16, numpy.float32, numpy.float64],
    }),
))
@testing.fix_random()
@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
        {'use_ideep': 'always'},
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestSoftmax(testing.FunctionTestCase):

    dodge_nondifferentiable = True

    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_options.update({'atol': 1e-3, 'rtol': 1e-2})
            self.check_backward_options.update({'atol': 1e-3, 'rtol': 1e-2})
            self.check_double_backward_options \
                .update({'atol': 1e-3, 'rtol': 1e-2})

    def generate_inputs(self):
        if self.shape is None:
            # For checking numerical stability
            value = -5 if self.dtype == numpy.float16 else -1000
            x = numpy.array([[value, 1]], dtype=self.dtype)
        else:
            x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        return functions.softmax(x, axis=self.axis),

    def forward_expected(self, inputs):
        x, = inputs
        expected = numpy.exp(x)
        expected = numpy.rollaxis(expected, self.axis, expected.ndim)
        for i in numpy.ndindex(expected.shape[:-1]):
            expected[i] /= expected[i].sum()
        expected = numpy.rollaxis(expected, expected.ndim-1, self.axis)
        return expected.astype(x.dtype),


@testing.parameterize(*testing.product({
    'axis': [0],
    'use_cudnn': ['always', 'auto', 'never'],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestSoftmaxCudnnCall(unittest.TestCase):

    def setUp(self):
        self.x = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        self.gy = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        with chainer.using_config('use_cudnn', self.use_cudnn):
            self.expect = chainer.should_use_cudnn('>=auto')

    def forward(self):
        x = chainer.Variable(self.x)
        return functions.softmax(x, axis=self.axis)

    def test_call_cudnn_forward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with testing.patch('cupy.cudnn.softmax_forward') as func:
                self.forward()
                self.assertEqual(func.called, self.expect)

    def test_call_cudnn_backward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            y = self.forward()
            y.grad = self.gy
            with testing.patch('cupy.cudnn.softmax_backward') as func:
                y.backward()
                self.assertEqual(func.called, self.expect)


testing.run_module(__name__, __file__)
