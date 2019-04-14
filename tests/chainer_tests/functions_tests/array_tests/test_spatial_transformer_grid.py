import numpy

from chainer import functions
from chainer import testing


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'use_cudnn': ['never', 'always'],
}))
@testing.inject_backend_tests(
    None,
    # CPU tests
    [
        {},
    ]
    # GPU tests
    + testing.product({
        'use_cuda': [True],
        'cuda_device': [0, 1],
        'use_cudnn': ['always'],
    })
    # ChainerX tests
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ])
class TestSpatialTransformerGrid(testing.FunctionTestCase):

    def setUp(self):
        self.skip_double_backward_test = True
        self.output_shape = (5, 6)

        if self.dtype == numpy.float16:
            self.check_forward_options = {'atol': 1e-3, 'rtol': 1e-3}
            self.check_backward_options = {'atol': 1e-2, 'rtol': 1e-1}
        else:
            self.check_forward_options = {}
            self.check_backward_options = {}

    def generate_inputs(self):
        theta = numpy.random.uniform(size=(3, 2, 3)).astype(self.dtype)
        return theta,

    def forward_expected(self, inputs):
        theta, = inputs
        B = theta.shape[0]
        H, W = self.output_shape

        expected = []
        for b in range(B):
            for i in numpy.linspace(-1., 1., H):
                for j in numpy.linspace(-1., 1., W):
                    coord = numpy.array([j, i, 1])
                    expected.append(theta[b].dot(coord))
        expected = numpy.array(
            expected).reshape(B, H, W, 2).transpose(0, 3, 1, 2)
        return expected.astype(self.dtype),

    def forward(self, inputs, device):
        theta, = inputs
        grid = functions.spatial_transformer_grid(theta, self.output_shape)
        return grid,


testing.run_module(__name__, __file__)
