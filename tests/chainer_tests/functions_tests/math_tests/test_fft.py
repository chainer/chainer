import numpy

from chainer import testing
from chainer import functions


@testing.parameterize(*testing.product({
    'shape': [(4,), (2, 3), (2, 3, 2)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'method': ['fft', 'ifft']
}))
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
        'cuda_device': [0, 1],
    })
    # ChainerX tests
    + testing.product({
        'use_chainerx': [True],
        'chainerx_device': ['native:0', 'cuda:0', 'cuda:1'],
    })
)
class TestFFT(testing.FunctionTestCase):

    def setUp(self):
        self.check_backward_options.update({
            'eps': 2.0 ** -2, 'atol': 1e-2, 'rtol': 1e-3})
        self.check_double_backward_options.update({
            'atol': 1e-2, 'rtol': 1e-3})

    def generate_inputs(self):
        rx = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        ix = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return rx, ix

    def forward(self, inputs, device):
        rx, ix = inputs
        ry, iy = getattr(functions, self.method)((rx, ix))
        return ry, iy

    def forward_expected(self, inputs):
        rx, ix = inputs
        expected = getattr(numpy.fft, self.method)(rx + ix * 1j)
        return (
            expected.real.astype(self.dtype),
            expected.imag.astype(self.dtype)
        )


testing.run_module(__name__, __file__)
