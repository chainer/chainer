import unittest

import numpy

import chainer
from chainer import functions
from chainer import testing
from chainer.utils import type_check


@testing.parameterize(*testing.product_dict(
    [{'dtype': numpy.float16},
     {'dtype': numpy.float32},
     {'dtype': numpy.float64},
     ],
    [{'slices': (0, slice(0, 1), numpy.array(-1)), 'b_data': numpy.array([1])},
     {'slices': (slice(None), 0, [0, 2]),
      'b_data': numpy.random.uniform(size=(4, 2))},
     {'slices': ([1, 0], [0, 0], [2, 0]),
      'b_data': numpy.random.uniform(size=(2,))},
     {'slices': 1, 'b_data': numpy.random.uniform(size=(2, 3))},
     {'slices': numpy.array([False, True, False, True]),
      'b_data': numpy.random.uniform(size=(2, 2, 3))},
     {'slices': [], 'b_data': numpy.empty(shape=(0, 2, 3))},
     ]
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
class TestScatterAdd(testing.FunctionTestCase):

    def setUp(self):
        self.check_backward_options.update({'atol': 5e-4, 'rtol': 5e-4})
        self.check_double_backward_options.update({
            'atol': 1e-3, 'rtol': 1e-3})

    def generate_inputs(self):
        a = numpy.random.uniform(-1, 1, (4, 2, 3)).astype(self.dtype)
        b = self.b_data.astype(self.dtype)
        return a, b

    def forward(self, inputs, device):
        a, b = inputs
        y = functions.scatter_add(a, self.slices, b)
        return y,

    def forward_expected(self, inputs):
        a, b = inputs
        a_copy = a.copy()
        numpy.add.at(a_copy, self.slices, b)
        return a_copy,


class TestInvalidScatterAdd(unittest.TestCase):

    def setUp(self):
        self.default_debug = chainer.is_debug()
        chainer.set_debug(True)

        self.a_data = numpy.random.uniform(-1, 1, (4, 3, 2))
        self.b_data = numpy.random.uniform(-1, 1, (2, 2))

    def tearDown(self):
        chainer.set_debug(self.default_debug)

    def test_multiple_ellipsis(self):
        with self.assertRaises(ValueError):
            functions.scatter_add(
                self.a_data, (Ellipsis, Ellipsis), self.b_data)

    def test_too_many_indices(self):
        with self.assertRaises(type_check.InvalidType):
            functions.scatter_add(self.a_data, (0, 0, 0, 0), self.b_data)

    def test_requires_broadcasting(self):
        with self.assertRaises(ValueError):
            functions.scatter_add(self.a_data, slice(0, 2), self.b_data)


testing.run_module(__name__, __file__)
