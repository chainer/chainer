import numpy
import pytest

import chainer
import chainerx
import chainerx.testing

from chainerx_tests import array_utils
from chainerx_tests import dtype_utils
from chainerx_tests import op_utils


def _skip_if_native_and_lapack_unavailable(device):
    if (device.backend.name == 'native'
            and not chainerx.linalg._is_lapack_available()):
        pytest.skip('LAPACK is not linked to ChainerX')


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
        super().setup()
        device = chainerx.get_default_device()

        _skip_if_native_and_lapack_unavailable(device)

        self.check_forward_options.update({'rtol': 1e-4, 'atol': 1e-4})
        self.check_backward_options.update({'rtol': 5e-3})
        self.check_double_backward_options.update({'rtol': 5e-3})


_numpy_does_not_support_0d_input113 = \
    numpy.lib.NumpyVersion(numpy.__version__) < '1.13.0'


_numpy_does_not_support_0d_input116 = \
    numpy.lib.NumpyVersion(numpy.__version__) < '1.16.0'


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(0, 0), (1, 1), (3, 3)],
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


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('shape', [(2, 3), (3, 2)])
def test_solve_invalid_shape(device, shape):
    _skip_if_native_and_lapack_unavailable(device)

    a = array_utils.create_dummy_ndarray(chainerx, shape, 'float32')
    b = array_utils.create_dummy_ndarray(chainerx, shape, 'float32')
    with pytest.raises(chainerx.DimensionError):
        chainerx.linalg.solve(a, b)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_solve_invalid_dtype(device):
    _skip_if_native_and_lapack_unavailable(device)

    a = array_utils.create_dummy_ndarray(chainerx, (3, 3), 'float16')
    b = array_utils.create_dummy_ndarray(chainerx, (3, 3), 'float16')
    with pytest.raises(chainerx.DtypeError):
        chainerx.linalg.solve(a, b)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(0, 0), (1, 1), (3, 3)],
        'dtype': ['float32', 'float64']
    })
))
class TestInverse(NumpyLinalgOpTest):

    # For zero sized input strides are different
    check_numpy_strides_compliance = False

    def generate_inputs(self):
        a = numpy.random.random(self.shape).astype(self.dtype)
        a = a * 10 + numpy.ones(self.shape)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        out = xp.linalg.inv(a)
        return out,


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('shape', [(2, 3), (3, 2)])
def test_inv_invalid_shape(device, shape):
    _skip_if_native_and_lapack_unavailable(device)

    a = array_utils.create_dummy_ndarray(chainerx, shape, 'float32')
    with pytest.raises(chainerx.DimensionError):
        chainerx.linalg.inv(a)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_inv_invalid_dtype(device):
    _skip_if_native_and_lapack_unavailable(device)

    a = array_utils.create_dummy_ndarray(chainerx, (3, 3), 'float16')
    with pytest.raises(chainerx.DtypeError):
        chainerx.linalg.inv(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(0, 0), (0, 3), (3, 0), (1, 1), (2, 3), (3, 2), (3, 3)],
        'dtype': ['float32', 'float64'],
        'full_matrices': [False],
        'compute_uv': [True]
    }) + chainer.testing.product({
        'shape': [(0, 0), (0, 3), (3, 0), (1, 1), (2, 3), (3, 2), (3, 3)],
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


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_svd_invalid_shape(device):
    _skip_if_native_and_lapack_unavailable(device)

    a = array_utils.create_dummy_ndarray(chainerx, (), 'float32')
    with pytest.raises(chainerx.DimensionError):
        chainerx.linalg.svd(a)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_svd_invalid_dtype(device):
    _skip_if_native_and_lapack_unavailable(device)

    a = array_utils.create_dummy_ndarray(chainerx, (2, 3), 'float16')
    with pytest.raises(chainerx.DtypeError):
        chainerx.linalg.svd(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(0, 0), (0, 3), (3, 0), (1, 1), (2, 3), (3, 2), (3, 3)],
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


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_pinv_invalid_shape(device):
    _skip_if_native_and_lapack_unavailable(device)

    a = array_utils.create_dummy_ndarray(chainerx, (), 'float32')
    with pytest.raises(chainerx.DimensionError):
        chainerx.linalg.pinv(a)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_pinv_invalid_dtype(device):
    _skip_if_native_and_lapack_unavailable(device)

    a = array_utils.create_dummy_ndarray(chainerx, (2, 3), 'float16')
    with pytest.raises(chainerx.DtypeError):
        chainerx.linalg.pinv(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # backward for 'r', 'raw' modes is not implemented
    chainer.testing.product({
        'shape': [(0, 3), (3, 0), (1, 1), (2, 3), (3, 2), (3, 3)],
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
        'shape': [(1, 1), (3, 3)],
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


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_qr_invalid_mode(device):
    _skip_if_native_and_lapack_unavailable(device)

    a = array_utils.create_dummy_ndarray(chainerx, (), 'float32')
    with pytest.raises(ValueError):
        chainerx.linalg.qr(a, mode='bad_mode')


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_qr_invalid_shape(device):
    _skip_if_native_and_lapack_unavailable(device)

    a = array_utils.create_dummy_ndarray(chainerx, (), 'float32')
    with pytest.raises(chainerx.DimensionError):
        chainerx.linalg.qr(a)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_qr_invalid_dtype(device):
    _skip_if_native_and_lapack_unavailable(device)

    a = array_utils.create_dummy_ndarray(chainerx, (2, 3), 'float16')
    with pytest.raises(chainerx.DtypeError):
        chainerx.linalg.qr(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(0, 0), (1, 1), (3, 3)],
        'in_dtypes': ['float32', 'float64']
    })
))
class TestCholesky(op_utils.NumpyOpTest):

    # For input with shape (0, 0) strides are different
    check_numpy_strides_compliance = False
    dodge_nondifferentiable = True

    def setup(self):
        device = chainerx.get_default_device()

        _skip_if_native_and_lapack_unavailable(device)

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


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_cholesky_invalid_not_positive_definite(device):
    _skip_if_native_and_lapack_unavailable(device)

    while True:
        a = numpy.random.random((3, 3)).astype('float32')
        try:
            numpy.linalg.cholesky(a)
        except numpy.linalg.LinAlgError:
            break
    a = chainerx.array(a)
    with pytest.raises(chainerx.ChainerxError):
        chainerx.linalg.cholesky(a)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_cholesky_invalid_semidefinite(device):
    _skip_if_native_and_lapack_unavailable(device)

    a = chainerx.array([[1, -2], [-2, 1]], dtype='float32')
    with pytest.raises(chainerx.ChainerxError):
        chainerx.linalg.cholesky(a)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_cholesky_invalid_shape(device):
    _skip_if_native_and_lapack_unavailable(device)

    a = array_utils.create_dummy_ndarray(chainerx, (2, 3), 'float32')
    with pytest.raises(chainerx.DimensionError):
        chainerx.linalg.cholesky(a)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_cholesky_invalid_dtype(device):
    _skip_if_native_and_lapack_unavailable(device)

    a = array_utils.create_dummy_ndarray(chainerx, (3, 3), 'float16')
    with pytest.raises(chainerx.DtypeError):
        chainerx.linalg.cholesky(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(0, 0), (1, 1), (3, 3)],
        'in_dtypes': ['float32', 'float64'],
        'UPLO': ['u', 'L']
    })
))
class TestEigh(NumpyLinalgOpTest):

    def setup(self):
        device = chainerx.get_default_device()

        _skip_if_native_and_lapack_unavailable(device)

        self.check_backward_options.update({
            'eps': 1e-5, 'rtol': 1e-3, 'atol': 1e-3})
        self.check_double_backward_options.update({
            'eps': 1e-5, 'rtol': 1e-3, 'atol': 1e-3})

    def generate_inputs(self):
        a = numpy.random.random(self.shape).astype(self.in_dtypes)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs

        if (_numpy_does_not_support_0d_input113 and a.size == 0):
            pytest.skip('Older NumPy versions do not work with empty arrays')

        # Input has to be symmetrized for backward test to work
        a = (a + a.T)/2. + 1e-3 * xp.eye(*self.shape)

        w, v = xp.linalg.eigh(a, UPLO=self.UPLO)

        # The sign of eigenvectors is not unique,
        # therefore absolute values are compared
        return w, xp.abs(v)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_eigh_invalid_uplo_type(device):
    _skip_if_native_and_lapack_unavailable(device)

    a = array_utils.create_dummy_ndarray(chainerx, (3, 3), 'float32')
    with pytest.raises(TypeError):
        chainerx.linalg.eigh(a, UPLO=None)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_eigh_invalid_uplo_value(device):
    _skip_if_native_and_lapack_unavailable(device)

    # TODO(hvy): Update the test when the error types are unified to either.
    a = array_utils.create_dummy_ndarray(chainerx, (3, 3), 'float32')
    with pytest.raises(ValueError):
        chainerx.linalg.eigh(a, UPLO='bad_UPLO')
    a = array_utils.create_dummy_ndarray(chainerx, (3, 3), 'float32')
    with pytest.raises(chainerx.ChainerxError):
        chainerx.linalg.eigh(a, UPLO='A')


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_eigh_invalid_shape(device):
    _skip_if_native_and_lapack_unavailable(device)

    a = array_utils.create_dummy_ndarray(chainerx, (2, 3), 'float32')
    with pytest.raises(chainerx.DimensionError):
        chainerx.linalg.eigh(a)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_eigh_invalid_dtype(device):
    _skip_if_native_and_lapack_unavailable(device)

    a = array_utils.create_dummy_ndarray(chainerx, (3, 3), 'float16')
    with pytest.raises(chainerx.DtypeError):
        chainerx.linalg.eigh(a)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product({
        'shape': [(0, 0), (1, 1), (3, 3)],
        'in_dtypes': ['float32', 'float64'],
        'UPLO': ['u', 'L'],
        'skip_backward_test': [True],
        'skip_double_backward_test': [True]
    })
))
class TestEigvalsh(NumpyLinalgOpTest):

    def generate_inputs(self):
        a = numpy.random.random(self.shape).astype(self.in_dtypes)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs

        if (_numpy_does_not_support_0d_input113 and a.size == 0):
            pytest.skip('Older NumPy versions do not work with empty arrays')

        w = xp.linalg.eigvalsh(a, UPLO=self.UPLO)
        return w,


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_eigvalsh_invalid_uplo_type(device):
    _skip_if_native_and_lapack_unavailable(device)

    a = array_utils.create_dummy_ndarray(chainerx, (3, 3), 'float32')
    with pytest.raises(TypeError):
        chainerx.linalg.eigvalsh(a, UPLO=None)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_eigvalsh_invalid_uplo_value(device):
    _skip_if_native_and_lapack_unavailable(device)

    # TODO(hvy): Update the test when the error types are unified to either.
    a = array_utils.create_dummy_ndarray(chainerx, (3, 3), 'float32')
    with pytest.raises(ValueError):
        chainerx.linalg.eigvalsh(a, UPLO='wrong')
    a = array_utils.create_dummy_ndarray(chainerx, (3, 3), 'float32')
    with pytest.raises(chainerx.ChainerxError):
        chainerx.linalg.eigvalsh(a, UPLO='A')


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_eigvalsh_invalid_shape(device):
    _skip_if_native_and_lapack_unavailable(device)

    a = array_utils.create_dummy_ndarray(chainerx, (2, 3), 'float32')
    with pytest.raises(chainerx.DimensionError):
        chainerx.linalg.eigvalsh(a)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_eigvalsh_invalid_dtype(device):
    _skip_if_native_and_lapack_unavailable(device)

    a = array_utils.create_dummy_ndarray(chainerx, (3, 3), 'float16')
    with pytest.raises(chainerx.DtypeError):
        chainerx.linalg.eigvalsh(a)
