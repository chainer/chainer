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
    ((0, 2), (2, 0)),
    ((2, 0), (0, 3)),
    ((0, 0), (0, 0)),
    ((2, 3), (3, 4)),
    ((1, 2, 3), (3, 4)),
    ((1, 2, 0), (0, 4)),
    ((1, 0, 3), (3, 0)),
    ((1, 0, 3), (3, 4)),
    ((1, 2, 3), (3, 0)),
    ((1, 2), (1, 2, 3)),
    ((1, 0), (1, 0, 3)),
    ((0, 2), (1, 2, 0)),
    ((0, 2), (1, 2, 3)),
    ((1, 2), (1, 2, 0)),
    ((4, 5, 2), (3, 2, 5)),
    ((2, 3, 4, 4), (3, 4, 2)),
    ((2, 2, 3, 1), (2, 1, 3, 1,  4)),
    ((2, 4, 3), (1, 2, 3, 2)),
    ((1, 2, 3, 0), (4, 0, 5)),
    ((1, 2, 0, 3), (4, 3, 0)),
    ((1, 2, 0, 3), (4, 3, 5))
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
        self.check_backward_options.update({'rtol': 5e-3})
        self.check_double_backward_options.update({'rtol': 5e-3})


_numpy_does_not_support_0d_input113 = \
    numpy.lib.NumpyVersion(numpy.__version__) < '1.13.0'


_numpy_does_not_support_0d_input116 = \
    numpy.lib.NumpyVersion(numpy.__version__) < '1.16.0'


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(0, 0), (1, 1), (3, 3), (6, 6)],
        'b_columns': [(), (1,), (3,), (4,)],
        'dtypes': [
            ('float32', 'float32'),
            ('float64', 'float64'),
            ('float64', 'float32'),
            ('float32', 'float64')]
    })
))
class TestSolve(NumpyLinalgOpTest):

    def generate_inputs(self):
        sv = numpy.random.uniform(1, 2, size=self.shape[0])
        a = chainer.testing.generate_matrix(
            self.shape, dtype=self.dtypes[0], singular_values=sv)
        b = numpy.random.random(
            (self.shape[0], *self.b_columns)).astype(self.dtypes[1])
        return a, b

    def forward_xp(self, inputs, xp):
        a, b = inputs
        out = xp.linalg.solve(a, b)
        return out,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(2, 3), (3, 2)],
        'dtype': ['float32', 'float64']
    })
))
class TestSolveFailing(NumpyLinalgOpTest):

    forward_accept_errors = (numpy.linalg.LinAlgError,
                             chainerx.DimensionError)

    def generate_inputs(self):
        a = numpy.random.random(self.shape).astype(self.dtype)
        b = numpy.random.random(self.shape).astype(self.dtype)
        return a, b

    def forward_xp(self, inputs, xp):
        a, b = inputs
        out = xp.linalg.solve(a, b)
        return out,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('shape', [(3, 3)])
@chainer.testing.parameterize_pytest('dtype', ['float16'])
class TestSolveDtypeFailing(NumpyLinalgOpTest):

    forward_accept_errors = (TypeError,
                             chainerx.DtypeError)

    def generate_inputs(self):
        a = numpy.random.random(self.shape).astype(self.dtype)
        b = numpy.random.random(self.shape).astype(self.dtype)
        return a, b

    def forward_xp(self, inputs, xp):
        a, b = inputs
        out = xp.linalg.solve(a, b)
        return out,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(0, 0), (1, 1), (3, 3), (6, 6)],
        'dtype': ['float32', 'float64']
    })
))
class TestInverse(NumpyLinalgOpTest):

    # For zero sized input strides are different
    check_numpy_strides_compliance = False

    def generate_inputs(self):
        a = numpy.random.random(self.shape).astype(self.dtype)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        out = xp.linalg.inv(a)
        return out,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(2, 3), (3, 2)],
        'dtype': ['float32', 'float64']
    })
))
class TestInverseFailing(NumpyLinalgOpTest):

    forward_accept_errors = (numpy.linalg.LinAlgError,
                             chainerx.DimensionError)

    def generate_inputs(self):
        a = numpy.random.random(self.shape).astype(self.dtype)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        out = xp.linalg.inv(a)
        return out,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('shape', [(3, 3)])
@chainer.testing.parameterize_pytest('dtype', ['float16'])
class TestInverseDtypeFailing(NumpyLinalgOpTest):

    forward_accept_errors = (TypeError,
                             chainerx.DtypeError)

    def generate_inputs(self):
        a = numpy.random.random(self.shape).astype(self.dtype)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        out = xp.linalg.inv(a)
        return out,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(0, 0), (0, 3), (3, 0), (1, 1), (2, 3), (3, 2), (6, 6)],
        'dtype': ['float32', 'float64'],
        'full_matrices': [False],
        'compute_uv': [True]
    }) + chainer.testing.product({
        'shape': [(0, 0), (0, 3), (3, 0), (1, 1), (2, 3), (3, 2), (6, 6)],
        'dtype': ['float32', 'float64'],
        'full_matrices': [True],
        'compute_uv': [False],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True],
    })
))
class TestSVD(NumpyLinalgOpTest):

    def generate_inputs(self):
        a = numpy.random.random(self.shape).astype(self.dtype)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs

        if (_numpy_does_not_support_0d_input116 and a.size == 0):
            pytest.skip('Older NumPy versions do not work with empty arrays')

        out = xp.linalg.svd(a,
                            full_matrices=self.full_matrices,
                            compute_uv=self.compute_uv)
        # NOTE: cuSOLVER's (CuPy's) and NumPy's outputs of u and v might
        # differ in signs, which is not a problem mathematically
        if self.compute_uv:
            u, s, v = out
            return xp.abs(u), s, xp.abs(v)
        else:
            s = out
            return s,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('shape', [(2, 3)])
@chainer.testing.parameterize_pytest('dtype', ['float16'])
class TestSVDDtypeFailing(NumpyLinalgOpTest):

    forward_accept_errors = (TypeError,
                             chainerx.DtypeError)

    def generate_inputs(self):
        a = numpy.random.random(self.shape).astype(self.dtype)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        out = xp.linalg.svd(a)
        u, s, v = out
        return xp.abs(u), s, xp.abs(v)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(0, 0), (0, 3), (3, 0), (1, 1), (2, 3), (3, 2), (6, 6)],
        'rcond': [1e-15, 0.3, 0.5, 0.6],
        'dtype': ['float32', 'float64']
    })
))
class TestPseudoInverse(NumpyLinalgOpTest):

    # For zero sized input strides are different
    check_numpy_strides_compliance = False

    def generate_inputs(self):
        a = numpy.random.random(self.shape).astype(self.dtype)
        a = a * 10 + numpy.ones(self.shape)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs

        if (_numpy_does_not_support_0d_input113 and a.size == 0):
            pytest.skip('Older NumPy versions do not work with empty arrays')

        out = xp.linalg.pinv(a, rcond=self.rcond)
        return out,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(), ],
        'rcond': [1e-15, ],
        'dtype': ['float32', 'float64']
    })
))
class TestPseudoInverseFailing(NumpyLinalgOpTest):

    forward_accept_errors = (numpy.linalg.LinAlgError,
                             chainerx.ChainerxError,
                             chainerx.DimensionError)

    def generate_inputs(self):
        a = numpy.random.random(self.shape).astype(self.dtype)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        out = xp.linalg.pinv(a, rcond=self.rcond)
        return out,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('shape', [(2, 3)])
@chainer.testing.parameterize_pytest('dtype', ['float16'])
class TestPseudoInverseDtypeFailing(NumpyLinalgOpTest):

    forward_accept_errors = (TypeError,
                             chainerx.DtypeError)

    def generate_inputs(self):
        a = numpy.random.random(self.shape).astype(self.dtype)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        out = xp.linalg.pinv(a)
        return out,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # backward for 'r', 'raw' modes is not implemented
    chainer.testing.product({
        'shape': [(0, 3), (3, 0), (1, 1), (2, 3), (3, 2), (6, 6)],
        'in_dtypes': ['float32', 'float64'],
        'mode': ['r', 'raw'],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True]
    }) +
    # backward for non-square `R` is not implemented
    chainer.testing.product({
        'shape': [(0, 3), (3, 0), (2, 3), (3, 2)],
        'in_dtypes': ['float32', 'float64'],
        'mode': ['complete', 'reduced'],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True]
    }) +
    chainer.testing.product({
        'shape': [(1, 1), (6, 6)],
        'in_dtypes': ['float32', 'float64'],
        'mode': ['reduced', 'complete']
    }) + chainer.testing.product({
        'shape': [(3, 2)],
        'in_dtypes': ['float32', 'float64'],
        'mode': ['reduced']
    })
))
class TestQR(NumpyLinalgOpTest):

    # For input with shape (N, 0) strides are different
    check_numpy_strides_compliance = False

    def generate_inputs(self):
        singular_values = numpy.random.uniform(
            low=0.1, high=1.5, size=min(self.shape))
        a = chainer.testing.generate_matrix(
            self.shape, self.in_dtypes, singular_values=singular_values)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        if (numpy.lib.NumpyVersion(numpy.__version__) < '1.16.0'
                and a.size == 0):
            pytest.skip('Older NumPy versions do not work with empty arrays')
        out = xp.linalg.qr(a, mode=self.mode)

        if self.mode == 'r':
            r = out
            return r,
        if self.mode == 'raw':
            if a.dtype.char == 'f':
                return out[0].astype(xp.float64), out[1].astype(xp.float64)
        return out


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(1, 1), (2, 3), (3, 2), (6, 6)],
        'in_dtypes': ['float16'],
        'mode': ['r', 'raw', 'reduced', 'complete']
    })
))
class TestQRFailing(NumpyLinalgOpTest):

    forward_accept_errors = (TypeError,
                             chainerx.DtypeError)

    def generate_inputs(self):
        a = numpy.random.random(self.shape).astype(self.in_dtypes)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        out = xp.linalg.qr(a, mode=self.mode)
        return out


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(0, 0), (1, 1), (3, 3), (6, 6)],
        'in_dtypes': ['float32', 'float64']
    })
))
class TestCholesky(op_utils.NumpyOpTest):

    # For input with shape (0, 0) strides are different
    check_numpy_strides_compliance = False
    dodge_nondifferentiable = True

    def setup(self):
        device = chainerx.get_default_device()
        if (device.backend.name == 'native'
                and not chainerx.linalg._is_lapack_available()):
            pytest.skip('LAPACK is not linked to ChainerX')
        self.check_backward_options.update({
            'eps': 1e-5, 'rtol': 1e-4, 'atol': 1e-4})
        self.check_double_backward_options.update({
            'eps': 1e-5, 'rtol': 1e-4, 'atol': 1e-4})

    def generate_inputs(self):
        a = numpy.random.random(self.shape).astype(self.in_dtypes)
        # Make random square matrix a symmetric positive definite one
        a = numpy.array(a.T.dot(a)) + 1e-3 * numpy.eye(*self.shape)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs

        if (_numpy_does_not_support_0d_input113 and a.size == 0):
            pytest.skip('Older NumPy versions do not work with empty arrays')

        # Input has to be symmetrized for backward test to work
        a = (a + a.T)/2. + 1e-3 * xp.eye(*self.shape)

        L = xp.linalg.cholesky(a)
        return L,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(), (2, 3), (3, 2), (6, 6)],
        'in_dtypes': ['float32', 'float64'],
    })
))
class TestCholeskyFailing(NumpyLinalgOpTest):

    forward_accept_errors = (numpy.linalg.LinAlgError,
                             chainerx.ChainerxError,
                             chainerx.DimensionError)

    def generate_inputs(self):
        a = numpy.random.random(self.shape).astype(self.in_dtypes)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        L = xp.linalg.cholesky(a)
        return L,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'in_dtypes': ['float32', 'float64'],
    })
))
class TestCholeskySemiDefiniteFailing(NumpyLinalgOpTest):

    forward_accept_errors = (numpy.linalg.LinAlgError,
                             chainerx.ChainerxError)

    def generate_inputs(self):
        a = numpy.array([[1, -2], [-2, 1]]).astype(self.in_dtypes)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        L = xp.linalg.cholesky(a)
        return L,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(6, 6)],
        'in_dtypes': ['float16'],
    })
))
class TestCholeskyDtypeFailing(NumpyLinalgOpTest):

    forward_accept_errors = (TypeError,
                             chainerx.DtypeError)

    def generate_inputs(self):
        a = numpy.random.random(self.shape).astype(self.in_dtypes)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        L = xp.linalg.cholesky(a)
        return L,
