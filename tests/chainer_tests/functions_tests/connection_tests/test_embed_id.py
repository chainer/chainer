import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer.functions.connection import embed_id
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import backend


@testing.parameterize(*testing.product_dict(
    [
        {'x': (0, 1, 2), 'w_shape': (5, 3), 'ignore_label': 1},
        {'x': (2, 1, 2), 'w_shape': (6, 3), 'ignore_label': None},
        {'x': (3, 1), 'w_shape': (4, 3), 'ignore_label': 3},
    ], [
        {'w_dtype': numpy.float16},
        {'w_dtype': numpy.float32},
        {'w_dtype': numpy.float64},
    ], [
        {'x_dtype': numpy.int16},
        {'x_dtype': numpy.int32},
        {'x_dtype': numpy.int64},
    ]
))
@testing.fix_random()
@backend.inject_backend_tests(
    None,
    # ChainerX tests
    testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0'],
    })
    # CPU tests
    + testing.product({
        'use_cuda': [False],
        'use_ideep': ['never', 'always'],
    })
    # GPU tests
    + testing.product([
        [{'use_cuda': True}],

        # Without cuDNN
        testing.product({
            'use_cudnn': ['never'],
        })
        # With cuDNN
        + testing.product({
            'use_cudnn': ['always'],
            'cudnn_deterministic': [True, False],
            'autotune': [True, False],
        })]))
class TestEmbedID(testing.FunctionTestCase):

    def setUp(self):
        self.check_backward_options = {'atol': 1e-2, 'rtol': 1e-2}
        self.check_double_backward_options = {'atol': 1e-2, 'rtol': 1e-2}

    def generate_inputs(self):
        x = numpy.array(self.x).astype(self.x_dtype)
        W = numpy.random.uniform(-1, 1, self.w_shape).astype(self.w_dtype)
        return x, W,

    def forward(self, inputs, device):
        x, W = inputs
        out = chainer.functions.embed_id(x, W, self.ignore_label)
        return out,

    def forward_expected(self, inputs):
        x, W = inputs
        if self.ignore_label is not None:
            mask = (x == self.ignore_label)
            return numpy.where(mask[..., None], 0, W[numpy.where(mask, 0, x)]),
        return W[x],


@testing.parameterize(
    {'x_data': [0, 1, 0], 'ignore_label': None},
    {'x_data': [[0, 1, 0], [1, 0, 1]], 'ignore_label': None},
    {'x_data': [0, 1, -1], 'ignore_label': -1},
    {'x_data': [[0, 1, -1], [-1, 0, 1]], 'ignore_label': -1},
    {'x_data': [0, 1, 2], 'ignore_label': 2},
    {'x_data': [[0, 1, 0], [1, 0, 1]], 'ignore_label': 1},
)
class TestEmbedIdGrad(unittest.TestCase):

    n_unit = (4,)
    w_shape = (4, 2)

    def setUp(self):
        self.x = numpy.array(self.x_data, dtype='i')
        self.gy = numpy.random.uniform(
            -1, 1, self.x.shape + (2,)).astype('f')
        self.ggW = numpy.random.uniform(-1, 1, self.w_shape).astype('f')

    def check_backward(self, x, gy, ggW):
        return

        def f(x, gy):
            emb = embed_id.EmbedIDGrad(
                self.w_shape, self.ignore_label)
            return emb.apply((x, numpy.zeros(()), gy))[0]

        gradient_check.check_backward(f, (x, gy), (ggW,))

    def test_backward_cpu(self):
        self.check_backward(self.x, self.gy, self.ggW)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x), cuda.to_gpu(self.gy), cuda.to_gpu(self.ggW))


testing.run_module(__name__, __file__)
