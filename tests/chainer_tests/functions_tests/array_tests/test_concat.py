import unittest

import numpy

from chainer import functions
from chainer import testing
from chainer.testing import backend


@backend.inject_backend_tests(
    None,
    # CPU tests
    testing.product({
        'use_cuda': [False],
        'use_ideep': ['never', 'always'],
    })
    # GPU tests
    + [{'use_cuda': True}]
    # ChainerX tests
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
    ])
@testing.parameterize(*testing.product_dict(
    [
        {'shape': (2, 7, 3), 'axis': 1,
         'slices': [(slice(None), slice(None, 2)), (slice(None), slice(2, 5)),
                    (slice(None), slice(5, None))]},
        {'shape': (7, 3), 'axis': 0,
         'slices': [slice(None, 2), slice(2, 5), slice(5, None)]},
        {'shape': (2,), 'axis': 0, 'slices': [slice(None, 1), slice(1, None)]},
        {'shape': (2,), 'axis': 0, 'slices': [()]},
        {'shape': (2, 7, 3), 'axis': 1,
         'slices': [(slice(None), slice(None, 2)), (slice(None), slice(2, 5)),
                    (slice(None), slice(5, None))]},
        {'shape': (2, 7, 3), 'axis': 1,
         'slices': [(slice(None), slice(None, 2)), (slice(None), slice(2, 5)),
                    (slice(None), slice(5, None))]},
        {'shape': (2, 7, 3), 'axis': -2,
         'slices': [(slice(None), slice(None, 2)), (slice(None), slice(2, 5)),
                    (slice(None), slice(5, None))]},
        {'shape': (7, 3, 2, 2), 'axis': 0,
         'slices': [slice(None, 2), slice(2, 5), slice(5, None)]},
        {'shape': (2, 7, 3, 5), 'axis': 1,
         'slices': [(slice(None), slice(None, 2), slice(None)),
                    (slice(None), slice(2, 5), slice(None)),
                    (slice(None), slice(5, None), slice(None))]},
        {'shape': (2, 7, 3, 5), 'axis': -1,
         'slices': [(slice(None), slice(None), slice(None), slice(None, 2)),
                    (slice(None), slice(None), slice(None), slice(2, 3)),
                    (slice(None), slice(None), slice(None), slice(3, None))]},
        {'shape': (2, 7, 3, 5), 'axis': -3,
         'slices': [(slice(None), slice(None, 2), slice(None)),
                    (slice(None), slice(2, 5), slice(None)),
                    (slice(None), slice(5, None), slice(None))]},
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
class TestConcat(testing.FunctionTestCase):

    def generate_inputs(self):
        shape = self.shape
        dtype = self.dtype
        y = numpy.random.uniform(-1, 1, shape).astype(dtype)
        xs = tuple([y[s] for s in self.slices])
        return xs

    def forward(self, inputs, device):
        y = functions.concat(inputs, self.axis)
        return y,

    def forward_expected(self, inputs):
        y = numpy.concatenate(inputs, self.axis)
        return y,


class TestConcatInvalidAxisType(unittest.TestCase):

    def test_invlaid_axis_type(self):
        inputs = [numpy.random.rand(3, 4), numpy.random.rand(3, 1)]

        with self.assertRaises(TypeError):
            functions.concat(inputs, 'a')


testing.run_module(__name__, __file__)
