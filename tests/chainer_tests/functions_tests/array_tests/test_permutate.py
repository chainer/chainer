import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.utils import force_array


@testing.parameterize(*testing.product_dict(
    [{'shape': (3,), 'dtype': 'f', 'axis': 0, 'inv': False},
     {'shape': (3,), 'dtype': 'f', 'axis': -1, 'inv': True},
     {'shape': (3, 4), 'dtype': 'd', 'axis': 1, 'inv': True},
     {'shape': (3, 4, 5), 'dtype': 'f', 'axis': 2, 'inv': False}],
    [{'label_dtype': numpy.int8},
     {'label_dtype': numpy.int16},
     {'label_dtype': numpy.int32},
     {'label_dtype': numpy.int64}]
))
@testing.fix_random()
@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
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
class TestPermutate(testing.FunctionTestCase):

    def setUp(self):
        self.skip_double_backward_test = True
        self.check_backward_options.update({'atol': 1e-3, 'rtol': 1e-3})

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        indices = numpy.random.permutation(
            self.shape[self.axis]).astype(self.label_dtype)
        return x, indices

    def forward(self, inputs, device):
        x, indices = inputs
        y = functions.permutate(x, indices, axis=self.axis, inv=self.inv)
        return y,

    def forward_expected(self, inputs):
        x, indices = inputs
        if self.inv:
            indices = numpy.argsort(indices)
        expected = numpy.take(x, indices, axis=self.axis)
        expected = force_array(expected)
        return expected,


@testing.parameterize(
    {'indices': [0, 0]},
    {'indices': [-1, 0]},
    {'indices': [0, 2]},
)
class TestPermutateInvalidIndices(unittest.TestCase):

    def setUp(self):
        self.x = numpy.arange(10).reshape((2, 5)).astype('f')
        self.ind = numpy.array(self.indices, 'i')
        self.debug = chainer.is_debug()
        chainer.set_debug(True)

    def tearDown(self):
        chainer.set_debug(self.debug)

    def check_invalid(self, x_data, ind_data):
        x = chainer.Variable(x_data)
        ind = chainer.Variable(ind_data)
        with self.assertRaises(ValueError):
            functions.permutate(x, ind)

    def test_invlaid_cpu(self):
        self.check_invalid(self.x, self.ind)

    @attr.gpu
    def test_invlaid_gpu(self):
        self.check_invalid(cuda.to_gpu(self.x), cuda.to_gpu(self.ind))


testing.run_module(__name__, __file__)
