import unittest

import numpy
import six

import chainer
from chainer import functions
from chainer import testing
from chainer.testing import backend


def inject_backend_tests():
    decorator = backend.inject_backend_tests(
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
            {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
        ])
    return decorator


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (2, 7, 3), 'axis': 1, 'ys_section': [2, 5],
         'slices': [
             (slice(None), slice(None, 2)),
             (slice(None), slice(2, 5)),
             (slice(None), slice(5, None))]},
        {'shape': (7, 3), 'axis': 0, 'ys_section': [2, 5],
         'slices': [slice(None, 2), slice(2, 5), slice(5, None)]},
        {'shape': (7, 0), 'axis': 0, 'ys_section': [2, 5],
         'slices': [slice(None, 2), slice(2, 5), slice(5, None)]},
        {'shape': (2, 9, 3), 'axis': 1, 'ys_section': 3,
         'slices': [
             (slice(None), slice(None, 3)),
             (slice(None), slice(3, 6)),
             (slice(None), slice(6, None))]},
        {'shape': (2, 6, 3), 'axis': 1, 'ys_section': 3,
         'slices': [
             (slice(None), slice(None, 2)),
             (slice(None), slice(2, 4)),
             (slice(None), slice(4, None))]},
        {'shape': (2,), 'axis': 0, 'ys_section': [1],
         'slices': [slice(None, 1), slice(1, None)]},
        {'shape': (2,), 'axis': 0, 'ys_section': [],
         'slices': [slice(None, None)]},
        {'shape': (2, 7, 3), 'axis': 1, 'ys_section': [2, 5],
         'slices': [
             (slice(None), slice(None, 2)),
             (slice(None), slice(2, 5)),
             (slice(None), slice(5, None))]},
        {'shape': (2, 7, 3), 'axis': 1, 'ys_section': [0],
         'slices': [
             (slice(None), slice(None, 0)),
             (slice(None), slice(0, 7))]
         },
        {'shape': (2, 7, 3), 'axis': 1, 'ys_section': [7],
         'slices': [
             (slice(None), slice(None, 7)),
             (slice(None), slice(7, 7))]
         },
        {'shape': (2, 7, 3, 2), 'axis': 1, 'ys_section': [2, 5],
         'slices': [
             (slice(None), slice(None, 2)),
             (slice(None), slice(2, 5)),
             (slice(None), slice(5, None))]},
        {'shape': (2, 7, 3, 2), 'axis': 1, 'ys_section': [0],
         'slices': [
             (slice(None), slice(None, 0)),
             (slice(None), slice(0, 7))]
         },
        {'shape': (10, 4, 3, 2), 'axis': 0, 'ys_section': 1,
         'slices': [slice(None, None)]
         },
        {'shape': (10, 4, 3, 2), 'axis': 0, 'ys_section': 2,
         'slices': [slice(None, 5), slice(5, None)]
         },
        {'shape': (10, 4, 3, 2), 'axis': 0, 'ys_section': [],
         'slices': [slice(None, None)]
         },
        {'shape': (10, 4, 3, 2), 'axis': 0, 'ys_section': [0, 5],
         'slices': [slice(0, 0), slice(0, 5), slice(5, None)]
         },
        {'shape': (10, 4, 3, 2), 'axis': 0, 'ys_section': [0, 0, 5],
         'slices': [slice(0, 0), slice(0, 0), slice(None, 5), slice(5, None)]
         },
        {'shape': (10, 4, 3, 2), 'axis': 0, 'ys_section': [2, 3, 5],
         'slices': [slice(None, 2), slice(2, 3), slice(3, 5), slice(5, None)]
         },
        {'shape': (10, 4, 3, 2), 'axis': 0,
         'ys_section': numpy.asarray([2, 3, 5]),
         'slices': [slice(None, 2), slice(2, 3), slice(3, 5), slice(5, None)]
         },
        {'shape': (10, 4, 3, 2), 'axis': 0, 'ys_section': [2, 3, 3, 5],
         'slices': [slice(None, 2), slice(2, 3), slice(3, 3), slice(3, 5),
                    slice(5, None)]
         },
        {'shape': (5, 5, 3, 8), 'axis': 3, 'ys_section': 2,
         'slices': [
             (slice(None, None), slice(None, None), slice(None, None),
              slice(None, 4)),
             (slice(None, None), slice(None, None), slice(None, None),
              slice(4, None))]
         },
        {'shape': (5, 8, 3, 2), 'axis': -3, 'ys_section': 2,
         'slices': [(slice(None, None), slice(None, 4)),
                    (slice(None, None), slice(4, None))]
         },
        {'shape': (5, 8, 3, 2), 'axis': 1, 'ys_section': 2,
         'slices': [(slice(None, None), slice(None, 4)),
                    (slice(None, None), slice(4, None))]
         },
        {'shape': (5, 4, 3, 4), 'axis': -1, 'ys_section': 2,
         'slices': [
             (slice(None, None), slice(None, None), slice(None, None),
              slice(None, 2)),
             (slice(None, None), slice(None, None), slice(None, None),
              slice(2, None))]
         },
        {'shape': (10, 4, 3, 2), 'axis': 0, 'ys_section': numpy.array([]),
         'slices': [slice(None, None)]
         },
        # Functions with multiple outputs may receive `None` upstream gradients
        # in their backward method, `split_axis` must handle this case
        # (by constructing 0-filled variables for `None` gradients).
        {'shape': (2, 7, 3), 'axis': 1, 'ys_section': [2, 5],
         'slices': [
             (slice(None), slice(None, 2)),
             (slice(None), slice(2, 5)),
             (slice(None), slice(5, None))],
         'grad_outputs_is_none': [False, True, False]
         },
        {'shape': (2, 7, 3, 1), 'axis': 1, 'ys_section': [2, 5],
         'slices': [
             (slice(None), slice(None, 2)),
             (slice(None), slice(2, 5)),
             (slice(None), slice(5, None))],
         'grad_outputs_is_none': [False, True, False]
         },
    ],
    [
        {'dtype': numpy.float16},
        {'dtype': numpy.float32},
        {'dtype': numpy.float64},
    ],
))
@inject_backend_tests()
class TestSplitAxis(testing.FunctionTestCase):

    # A list of booleans. If element i is `True`, the i-th upstream gradient is
    # generated as `None`. Default is `None`, in which case all gradients are
    # ndarrays.
    grad_outputs_is_none = None

    def generate_inputs(self):
        shape = self.shape
        dtype = self.dtype
        x = numpy.arange(numpy.prod(shape), dtype=dtype).reshape(shape)
        return x,

    def generate_grad_outputs(self, outputs_template):
        grad_outputs = tuple([
            numpy.random.uniform(-1, 1, a.shape).astype(a.dtype)
            for a in outputs_template])

        if self.grad_outputs_is_none is not None:
            assert len(self.grad_outputs_is_none) == len(grad_outputs)
            grad_outputs = tuple(
                None if is_none else g for is_none, g,
                in six.moves.zip(self.grad_outputs_is_none, grad_outputs))

        return grad_outputs

    def forward(self, inputs, device):
        x, = inputs
        return functions.split_axis(
            x, self.ys_section, self.axis, force_tuple=True)

    def forward_expected(self, inputs):
        x, = inputs
        return tuple([x[s] for s in self.slices])


@inject_backend_tests()
class TestSplitAxisNone(testing.FunctionTestCase):

    skip_double_backward_test = True

    axis = 0
    ys_section = [1]

    def generate_inputs(self):
        x = numpy.array([1, 2], dtype=numpy.float32)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        return functions.split_axis(
            x, self.ys_section, self.axis)

    def forward_expected(self, inputs):
        x, = inputs
        return tuple(numpy.split(x, self.ys_section, self.axis))


@testing.parameterize(
    {'force_tuple': True},
    {'force_tuple': False},
)
@inject_backend_tests()
class TestSplitAxisForceArray(testing.FunctionTestCase):

    skip_backward_test = True
    skip_double_backward_test = True

    axis = 1

    def generate_inputs(self):
        x = numpy.arange(42, dtype=numpy.float32).reshape(2, 7, 3)
        return x,

    def forward(self, inputs, device):
        x, = inputs
        ret = functions.split_axis(
            x, 1, self.axis, force_tuple=self.force_tuple)
        if self.force_tuple:
            assert isinstance(ret, tuple)
            assert len(ret) == 1
            return ret
        else:
            assert isinstance(ret, chainer.Variable)
            return ret,

    def forward_expected(self, inputs):
        x, = inputs
        return tuple(numpy.split(x, 1, self.axis))


class TestSplitAxisInvalidSections(unittest.TestCase):

    def setUp(self):
        self.default_debug = chainer.is_debug()
        chainer.set_debug(True)

    def tearDown(self):
        chainer.set_debug(self.default_debug)

    def test_invalid_sections(self):
        x = numpy.zeros((2, 3, 4), dtype='f')
        with self.assertRaises(ValueError):
            functions.split_axis(x, [2, 1], 1)


testing.run_module(__name__, __file__)
