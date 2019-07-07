import numpy
import pytest

import chainer
import chainerx
import chainerx.testing

from chainerx_tests import array_utils
from chainerx_tests import dtype_utils
from chainerx_tests import op_utils


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('a_shape,b_shape', [
    ((), ()),
    ((), (2, 3)),
    ((2, 0), (0, 3)),
    ((0, 0), (0, 0)),
    ((2, 3), (3, 4)),
    ((4, 5, 2), (3, 2, 5)),
    ((2, 3, 4, 4), (3, 4, 2)),
    ((2, 2, 3, 1), (2, 1, 3, 1,  4)),
    ((2, 4, 3), (1, 2, 3, 2))

    # TODO(niboshi): Add test cases for more than 2 ndim
])
@chainer.testing.parameterize_pytest(
    'in_dtypes,chx_expected_dtype', dtype_utils.result_dtypes_two_arrays)
@chainer.testing.parameterize_pytest('is_module', [True, False])
class TestDot(op_utils.NumpyOpTest):

    def setup(self):
        device = chainerx.get_default_device()
        a_dtype, b_dtype = self.in_dtypes
        a_kind = numpy.dtype(a_dtype).kind
        b_kind = numpy.dtype(b_dtype).kind
        # TODO(beam2d): Remove the skip after supporting non-float dot on CUDA
        if device.name == 'cuda:0' and (a_kind != 'f' and b_kind != 'f'):
            pytest.skip('non-float dot is not supported on CUDA')

        # Skip backward/double-backward tests for int dtypes
        if a_kind != 'f' or b_kind != 'f':
            self.skip_backward_test = True
            self.skip_double_backward_test = True
        # Skip backward/double-backward tests if the output will be
        # disconnected.
        # TODO(niboshi): Remove this skip condition after enabling backward()
        # for such cases.
        if self.a_shape and self.a_shape[-1] == 0:
            self.skip_backward_test = True
            self.skip_double_backward_test = True

        if a_dtype == 'float16' or b_dtype == 'float16':
            self.check_forward_options.update({
                'rtol': 1e-2, 'atol': 1e-2})
            self.check_backward_options.update({
                'rtol': 1e-2, 'atol': 1e-2})
            self.check_double_backward_options.update({
                'rtol': 1e-2, 'atol': 1e-2})

    def generate_inputs(self):
        a_dtype, b_dtype = self.in_dtypes
        a_shape = self.a_shape
        b_shape = self.b_shape
        a = numpy.random.uniform(-1, 1, a_shape).astype(a_dtype)
        b = numpy.random.uniform(-1, 1, b_shape).astype(b_dtype)
        return a, b

    def forward_xp(self, inputs, xp):
        a, b = inputs
        if self.is_module:
            y = xp.dot(a, b)
        else:
            y = a.dot(b)
        y = dtype_utils.cast_if_numpy_array(xp, y, self.chx_expected_dtype)
        return y,


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(chainerx.DimensionError, ValueError))
@pytest.mark.parametrize('a_shape,b_shape', [
    ((3, 2), (1, 3)),
    ((4, 3, 2, 5), (6, 4, 1, 2))
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_dot_invalid(is_module, xp, device, a_shape, b_shape, dtype):
    # TODO(beam2d): Remove the skip after supporting non-float dot on CUDA
    if device.name == 'cuda:0' and numpy.dtype(dtype).kind != 'f':
        return chainerx.testing.ignore()
    a = array_utils.create_dummy_ndarray(xp, a_shape, dtype)
    b = array_utils.create_dummy_ndarray(xp, b_shape, dtype)
    if is_module:
        return xp.dot(a, b)
    else:
        return a.dot(b)


class NumpyLinalgOpTest(op_utils.NumpyOpTest):

    dodge_nondifferentiable = True

    def setup(self):
        device = chainerx.get_default_device()
        if (device.backend.name == 'native'
                and not chainerx.linalg._is_lapack_available()):
            pytest.skip('LAPACK is not linked to ChainerX')


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(1, 1), (3, 3), (6, 6)],
        'in_dtypes': ['float32', 'float64'],
        'contiguous': [None, 'C'],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True]
    })
))
class TestCholesky(NumpyLinalgOpTest):

    def generate_inputs(self):
        a = numpy.random.random(self.shape).astype(self.in_dtypes)
        # Make random square matrix a symmetric positive semi-definite one
        a = numpy.array(a.T.dot(a))
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        L = xp.linalg.cholesky(a)
        return L,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # Special shapes
    chainer.testing.product({
        'shape': [(), (2, 3), (3, 2), (6, 6)],
        'in_dtypes': ['float16', 'float32', 'float64'],
        'contiguous': [None, 'C'],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True]
    })
))
class TestCholeskyFailing(NumpyLinalgOpTest):

    forward_accept_errors = (numpy.linalg.LinAlgError,
                             chainerx.ChainerxError,
                             chainerx.DimensionError,
                             TypeError,
                             chainerx.DtypeError)

    def generate_inputs(self):
        a = numpy.random.random(self.shape).astype(self.in_dtypes)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        L = xp.linalg.cholesky(a)
        return L,
