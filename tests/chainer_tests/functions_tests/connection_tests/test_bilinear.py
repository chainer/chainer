import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr


def _uniform(*shape):
    return numpy.random.uniform(-1, 1, shape).astype(numpy.float32)


@testing.parameterize(*testing.product({
    'in_shapes': [((2,), (4,)), ((2, 1), (4, 2))],
    'out_size': [3],
    'batch_size': [2],
    'test_partial': [True, False],
}))
@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
        {'use_ideep': ['never', 'always']},
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
class TestBilinearFunction(testing.FunctionTestCase):

    def setUp(self):
        self.e1_shape = (self.batch_size,) + self.in_shapes[0]
        self.e2_shape = (self.batch_size,) + self.in_shapes[1]
        self.e1_size = numpy.prod(self.in_shapes[0])
        self.e2_size = numpy.prod(self.in_shapes[1])

        self.check_backward_options = {
            'atol': 1e-5, 'rtol': 1e-4}
        self.check_double_backward_options = {
            'atol': 1e-4, 'rtol': 1e-3}

    def generate_inputs(self):
        e1 = _uniform(*self.e1_shape)
        e2 = _uniform(*self.e2_shape)
        W = _uniform(self.e1_size, self.e2_size, self.out_size)
        if self.test_partial:
            return e1, e2, W
        else:
            V1 = _uniform(self.e1_size, self.out_size)
            V2 = _uniform(self.e2_size, self.out_size)
            b = _uniform(self.out_size)

        return e1, e2, W, V1, V2, b

    def forward_expected(self, inputs):
        if self.test_partial:
            e1, e2, W = inputs
            V1 = None
            V2 = None
            b = None
        else:
            e1, e2, W, V1, V2, b = inputs

        e1 = e1.reshape(e1.shape[0], -1)
        e2 = e2.reshape(e2.shape[0], -1)
        xp = backend.get_array_module(e1)
        y_expect = xp.einsum('ij,ik,jkl->il', e1, e2, W)
        flags = V1 is None, V2 is None, b is None
        if any(flags):
            if not all(flags):
                raise ValueError(
                    'Test either all or none of the optional parameters.')
        else:
            y_expect += e1.dot(V1)
            y_expect += e2.dot(V2)
            y_expect += b
        return y_expect,

    def forward(self, inputs, device):
        if self.test_partial:
            e1, e2, W = inputs
            V1 = None
            V2 = None
            b = None
        else:
            e1, e2, W, V1, V2, b = inputs
        flags = V1 is None, V2 is None, b is None
        if any(flags):
            if not all(flags):
                raise ValueError(
                    'Test either all or none of the optional parameters.')
            y = functions.bilinear(e1, e2, W)
        else:
            y = functions.bilinear(e1, e2, W, V1, V2, b)
        return y,


@attr.slow
class TestBilinearFunctionLarge(unittest.TestCase):

    def setUp(self):
        self.e1 = _uniform(256, 256)
        self.e2 = _uniform(256, 256)
        self.w = _uniform(256, 256, 256)
        self.v1 = _uniform(256, 256)
        self.v2 = _uniform(256, 256)
        self.b = _uniform(256)

    def test_cpu(self):
        chainer.functions.bilinear(
            self.e1, self.e2, self.w, self.v1, self.v2, self.b)

    @attr.gpu
    def test_gpu(self):
        chainer.functions.bilinear(*map(cuda.to_gpu, (
            self.e1, self.e2, self.w, self.v1, self.v2, self.b)))


class TestBilinearFunctionInvalidArgument(unittest.TestCase):

    def setUp(self):
        e1 = _uniform(3, 2)
        e2 = _uniform(3, 4)
        W = _uniform(2, 4, 5)
        V1 = _uniform(2, 5)

        self.e1 = chainer.Variable(e1)
        self.e2 = chainer.Variable(e2)
        self.W = chainer.Variable(W)
        self.V1 = chainer.Variable(V1)

    def test_invalid_full_partial_ambiguous(self):
        with self.assertRaises(ValueError):
            functions.bilinear(self.e1, self.e2, self.W, self.V1)


testing.run_module(__name__, __file__)
