import unittest

import numpy

import chainer
from chainer import functions
from chainer import testing
from chainer.testing import attr
import chainerx


if chainerx.is_available():
    import chainerx.testing


@testing.parameterize(*testing.product_dict(
    [
        {'shape': (3, 4)},
        {'shape': ()},
    ],
    [
        {'in_type': numpy.bool_},
        {'in_type': numpy.uint8},
        {'in_type': numpy.uint64},
        {'in_type': numpy.int8},
        {'in_type': numpy.int64},
        {'in_type': numpy.float16},
        {'in_type': numpy.float32},
        {'in_type': numpy.float64},
    ],
    [
        {'out_type': numpy.bool_},
        {'out_type': numpy.uint8},
        {'out_type': numpy.uint64},
        {'out_type': numpy.int8},
        {'out_type': numpy.int64},
        {'out_type': numpy.float16},
        {'out_type': numpy.float32},
        {'out_type': numpy.float64},
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
    + [
        {'use_chainerx': True, 'chainerx_device': 'native:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:0'},
        {'use_chainerx': True, 'chainerx_device': 'cuda:1'},
    ]
)
@attr.chainerx
class TestCast(testing.FunctionTestCase):

    def _skip_chainerx_unsupported_dtype(self):
        supported_dtypes = chainerx.testing.dtypes.all_dtypes
        if (self.in_type.__name__ not in supported_dtypes
                or self.out_type.__name__ not in supported_dtypes):
            raise unittest.SkipTest(
                'ChainerX does not support either of {} or {} dtypes'.format(
                    self.in_type.__name__, self.out_type.__name__))

    def setUp(self):
        # Skip e.g. uint64 for ChainerX.
        self._skip_chainerx_unsupported_dtype()

        if (numpy.dtype(self.in_type).kind != 'f'
                or numpy.dtype(self.out_type).kind != 'f'):
            self.skip_backward_test = True
            self.skip_double_backward_test = True

        if (numpy.dtype(self.in_type).kind == 'f'
                and self.out_type == numpy.float16):
            self.check_forward_options.update({'atol': 1e-3, 'rtol': 1e-3})
        self.check_backward_options.update({
            'eps': 2.0 ** -2, 'atol': 1e-2, 'rtol': 1e-3})
        self.check_double_backward_options.update({
            'eps': 2.0 ** -2, 'atol': 1e-2, 'rtol': 1e-3})

    def generate_inputs(self):
        x = numpy.asarray(numpy.random.randn(*self.shape)).astype(self.in_type)
        # The result of a cast from a negative floating-point number to
        # an unsigned integer is not specified. Avoid testing that condition.
        float_to_uint = (
            issubclass(self.in_type, numpy.floating)
            and issubclass(self.out_type, numpy.unsignedinteger))
        if float_to_uint:
            x[x < 0] *= -1
        return x,

    def forward_expected(self, inputs):
        x, = inputs
        return x.astype(self.out_type),

    def forward(self, inputs, devices):
        x, = inputs
        y = functions.cast(x, self.out_type)
        return y,


class TestNoCast(unittest.TestCase):

    def setUp(self):
        self.dtype = numpy.float32
        self.x = numpy.random.uniform(-100, 100, (1,)).astype(self.dtype)

    def check_forward_no_cast(self, x_data):
        y = functions.cast(x_data, self.dtype)
        assert isinstance(y, chainer.Variable)
        assert y.data is x_data

    def test_forward_no_cast_array(self):
        y = functions.cast(self.x, self.dtype)
        assert isinstance(y, chainer.Variable)
        assert y.data is self.x

    def test_forward_no_cast_variable(self):
        # If backprop is disabled, it's safe to simply return the input
        # variable for no-op casts.
        x = chainer.Variable(self.x)
        with chainer.using_config('enable_backprop', False):
            y = functions.cast(x, self.dtype)
        assert y is x

    def test_forward_no_cast_grad(self):
        # This test would fail if F.cast does not create new function nodes for
        # no-op casts
        x = chainer.Variable(self.x)
        y1 = functions.cast(x, self.dtype)
        y2 = functions.cast(x, self.dtype)
        z = y1 + y2
        gy1, gy2 = chainer.grad([z], [y1, y2], [numpy.ones_like(z.data)])
        assert gy1.dtype == self.dtype
        assert gy2.dtype == self.dtype
        numpy.testing.assert_array_equal(gy1.data, numpy.ones_like(y1.data))
        numpy.testing.assert_array_equal(gy2.data, numpy.ones_like(y2.data))


testing.run_module(__name__, __file__)
