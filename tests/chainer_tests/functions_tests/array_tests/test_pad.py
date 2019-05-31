import numpy

from chainer import functions
from chainer import testing


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (), 'pad_width': 1, 'mode': 'constant'},
        {'shape': (2, 3), 'pad_width': 0, 'mode': 'constant'},
        {'shape': (2, 3), 'pad_width': 1, 'mode': 'constant'},
        {'shape': (2, 3), 'pad_width': (1, 2), 'mode': 'constant'},
        {'shape': (2, 3), 'pad_width': ((1, 2), (3, 4)), 'mode': 'constant'},
        {'shape': (2, 3, 2), 'pad_width': ((2, 5), (1, 2), (0, 7)),
         'mode': 'constant'},
        {'shape': (1, 3, 5, 2), 'pad_width': 2, 'mode': 'constant'}
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64}
    ]
))
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
class TestPadDefault(testing.FunctionTestCase):

    def setUp(self):
        self.check_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_backward_options.update({'atol': 3e-2, 'rtol': 3e-2})

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        y = functions.pad(x, self.pad_width, self.mode)
        return y,

    def forward_expected(self, inputs):
        x, = inputs
        y_expected = numpy.pad(x, self.pad_width, self.mode)
        return y_expected.astype(self.dtype),


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (2, 3), 'pad_width': 1, 'mode': 'constant',
         'constant_values': 1},
        {'shape': (2, 3), 'pad_width': (1, 2), 'mode': 'constant',
         'constant_values': (1, 2)},
        {'shape': (2, 3), 'pad_width': ((1, 2), (3, 4)), 'mode': 'constant',
         'constant_values': ((1, 2), (3, 4))},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64}
    ]
))
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
# Old numpy does not work with multi-dimensional constant_values
@testing.with_requires('numpy>=1.11.1')
class TestPad(testing.FunctionTestCase):

    def setUp(self):
        self.check_backward_options = {}
        if self.dtype == numpy.float16:
            self.check_backward_options.update({'atol': 3e-2, 'rtol': 3e-2})

    def generate_inputs(self):
        x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        return x,

    def forward_expected(self, inputs):
        x, = inputs
        y_expected = numpy.pad(x, self.pad_width, mode=self.mode,
                               constant_values=self.constant_values)
        return y_expected,

    def forward(self, inputs, device):
        x, = inputs
        y = functions.pad(x, self.pad_width, mode=self.mode,
                          constant_values=self.constant_values)
        return y,


testing.run_module(__name__, __file__)
