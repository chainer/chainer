import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'ratio': [0.1, 0.3, 0.5, 0.0],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
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
        'use_cudnn': ['never', 'always'],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1'],
    })
)
class TestDropout(testing.FunctionTestCase):
    def setUp(self):
        if self.dtype == numpy.float16:
            self.check_forward_options = ({'atol': 1e-3, 'rtol': 1e-2})
            self.check_backward_options = ({'atol': 1e-3, 'rtol': 1e-2})
            self.check_double_backward_options = ({'atol': 1e-3, 'rtol': 1e-2})
        dropout = functions.noise.dropout.Dropout(self.ratio)
        self.dropout = dropout

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        self.dropout.apply((x,))
        self.mask = self.dropout.mask
        return x,

    def forward(self, inputs, device):
        x, = inputs
        self.dropout.mask = device.send(self.mask)
        y, = self.dropout.apply((x,))
        return y,

    def forward_expected(self, inputs):
        x, = inputs
        if self.ratio == 0.0:
            y_expected = x
        else:
            y_expected = x * self.mask
        return y_expected,


@testing.parameterize(
    # The case where specify_mask is False and train is True is tested
    # in TestDropout, so is omitted.
    {'specify_mask': True, 'train': True},
    {'specify_mask': True, 'train': False},
    {'specify_mask': False, 'train': False},
)
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
        'use_cudnn': ['never', 'always'],
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1'],
    })
)
class TestDropoutMask(testing.FunctionTestCase):

    def setUp(self):
        self.skip_backward_test = True
        self.skip_double_backward_test = True

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, (2, 3)).astype(numpy.float32)
        self.mask = (numpy.random.uniform(-1, 1, (2, 3)) > 0).astype(
            numpy.float32)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        x.array = device.send(x.array)
        mask = device.send(self.mask) if self.specify_mask else None
        with chainer.using_config('train', self.train):
            y, y_mask = functions.dropout(x, 0.5, mask=mask,
                                          return_mask=True)
        if self.train:
            assert isinstance(y_mask, type(y.array))
            if mask is None:
                assert y_mask.shape == y.array.shape
            else:
                assert y_mask is mask
        else:
            assert y_mask is None
        return y,

    def forward_expected(self, inputs):
        x, = inputs
        if self.train is False:
            y_expected = x
        else:
            y_expected = x * self.mask
        return y_expected,


@testing.parameterize(*testing.product({
    'use_cudnn': ['never', 'always'],
    'dropout': [0, 0.5],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestDropoutCudnnCall(unittest.TestCase):

    def setUp(self):
        self.x = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)
        self.gy = cuda.cupy.random.uniform(-1, 1, (2, 3)).astype(self.dtype)

    def forward(self):
        return functions.dropout(chainer.Variable(self.x), self.dropout)

    def test_call_cudnn_forward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            with testing.patch(
                    'chainer.backends.cuda.get_cudnn_dropout_states') as func:
                self.forward()
                assert func.called == (self.use_cudnn == 'always')

    def test_call_cudnn_backward(self):
        with chainer.using_config('use_cudnn', self.use_cudnn):
            y = self.forward()
            y.grad = self.gy
            with testing.patch(
                    'chainer.backends.cuda.get_cudnn_dropout_states') as func:
                y.backward()
                assert func.called == (self.use_cudnn == 'always')


testing.run_module(__name__, __file__)
