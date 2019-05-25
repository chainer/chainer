import numpy

import chainer
from chainer.backends import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer import utils


@testing.parameterize(*testing.product_dict(
    [{'dtype': numpy.float16,
      'double_backward_options': {'atol': 3e-1, 'rtol': 3e-1}},
     {'dtype': numpy.float32,
      'double_backward_options': {}},
     {'dtype': numpy.float64,
      'double_backward_options': {}},
     ],
    [{'shape': (4, 3)},
     {'shape': (4, 3, 2)},
     {'shape': (4,)},
     {'shape': ()},
     {'shape': (1,)},
     {'shape': (1, 1)},
     ]
))
@testing.inject_backend_tests(
    None,
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
                {'atol': 5e-2, 'rtol': 5e-2})

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

    # test for #4669
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
