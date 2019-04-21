import unittest

import chainer
import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import array_utils
from chainerx_tests import dtype_utils
from chainerx_tests import op_utils


# Value for parameterization to represent an unspecified (default) argument.
class _UnspecifiedType(object):
    def __repr__(self):
        return '<Unspecified>'


_unspecified = _UnspecifiedType()


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('mode', ['module', 'transpose', 'T'])
class TestTranspose(op_utils.NumpyOpTest):

    def setup(self, shape, dtype):
        # Skip backward/double-backward tests for int dtypes
        if numpy.dtype(dtype).kind != 'f':
            self.skip_backward_test = True
            self.skip_double_backward_test = True
        self.shape = shape
        self.dtype = dtype

    def generate_inputs(self):
        shape = self.shape
        dtype = self.dtype
        a = array_utils.create_dummy_ndarray(numpy, shape, dtype)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        mode = self.mode
        if mode == 'module':
            b = xp.transpose(a)
        elif mode == 'transpose':
            b = a.transpose()
        elif mode == 'T':
            b = a.T
        else:
            assert False
        return b,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('shape,axes', [
    ((1,), 0),
    ((1,), (0,)),
    ((2,), (0,)),
    ((2, 3), (1, 0)),
    ((2, 3), (-2, -1)),
    ((2, 3, 1), (2, 0, 1)),
    ((2, 3, 1), (2, -3, 1)),
])
@chainer.testing.parameterize_pytest('is_module', [True, False])
class TestTransposeAxes(op_utils.NumpyOpTest):

    def setup(self, dtype):
        # Skip backward/double-backward tests for int dtypes
        if numpy.dtype(dtype).kind != 'f':
            self.skip_backward_test = True
            self.skip_double_backward_test = True

        self.dtype = dtype

    def generate_inputs(self):
        shape = self.shape
        dtype = self.dtype
        a = array_utils.create_dummy_ndarray(numpy, shape, dtype)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        axes = self.axes
        if self.is_module:
            b = xp.transpose(a, axes)
        else:
            b = a.transpose(axes)
        return b,


@pytest.mark.parametrize('shape,axes', [
    ((), (0,)),
    ((1,), (1,)),
    ((2, 3), (1,)),
    ((2, 3), (1, 0, 2)),
])
def test_transpose_invalid_axes(shape, axes):
    a = array_utils.create_dummy_ndarray(chainerx, shape, 'float32')
    with pytest.raises(chainerx.DimensionError):
        chainerx.transpose(a, axes)
    with pytest.raises(chainerx.DimensionError):
        a.transpose(axes)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('in_shape,axis,start', [
    # various axis
    ((2, 3, 4), 0, _unspecified),
    ((2, 3, 4), 1, _unspecified),
    ((2, 3, 4), 2, _unspecified),
    ((2, 3, 4), -1, _unspecified),
    ((2, 3, 4), -3, _unspecified),
    # with start
    ((2, 3, 4), 1, 0),
    ((2, 3, 4), 1, 1),
    ((2, 3, 4), 1, 2),
    ((2, 3, 4), 1, 3),
    ((2, 3, 4), 1, -1),
    ((2, 3, 4), 1, -2),
    ((2, 3, 4), 1, -3),
    ((2, 3, 4), 2, 3),
    ((2, 3, 4), 2, 0),
    ((2, 3, 4), 0, 3),
    ((2, 3, 4), 0, 0),
    # single dim
    ((1,), 0, _unspecified),
    ((1,), -1, _unspecified),
    # zero-length dims
    ((0,), 0, _unspecified),
    ((0,), 0, 0),
    ((0,), 0, 1),
    ((0,), -1, _unspecified),
    ((2, 0, 3), 1, _unspecified),
    ((2, 0, 3), -2, _unspecified),
])
class TestRollaxis(op_utils.NumpyOpTest):

    def setup(self, dtype):
        # Skip backward/double-backward tests for int dtypes
        if numpy.dtype(dtype).kind != 'f':
            self.skip_backward_test = True
            self.skip_double_backward_test = True

        self.dtype = dtype

    def generate_inputs(self):
        in_shape = self.in_shape
        dtype = self.dtype
        a = array_utils.create_dummy_ndarray(numpy, in_shape, dtype)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        axis = self.axis
        start = self.axis
        if start is _unspecified:
            b = xp.rollaxis(a, axis)
        else:
            b = xp.rollaxis(a, axis, start)
        return b,


@pytest.mark.parametrize('in_shape,axis,start', [
    # out of bounds axis
    ((2, 3, 4), 3, _unspecified),
    ((2, 3, 4), -4, _unspecified),
    # out of bounds start
    ((2, 3, 4), 2, 4),
    ((2, 3, 4), 2, -4),
    # empty shape
    ((), 0, _unspecified),
    ((), -1, _unspecified),
])
def test_rollaxis_invalid(in_shape, axis, start):
    a = array_utils.create_dummy_ndarray(chainerx, in_shape, 'float32')
    with pytest.raises(chainerx.DimensionError):
        if start is _unspecified:
            chainerx.rollaxis(a, axis)
        else:
            chainerx.rollaxis(a, axis, start)


_reshape_shape = [
    ((), ()),
    ((0,), (0,)),
    ((1,), (1,)),
    ((5,), (5,)),
    ((2, 3), (2, 3)),
    ((1,), ()),
    ((), (1,)),
    ((1, 1), ()),
    ((), (1, 1)),
    ((6,), (2, 3)),
    ((2, 3), (6,)),
    ((2, 0, 3), (5, 0, 7)),
    ((5,), (1, 1, 5, 1, 1)),
    ((1, 1, 5, 1, 1), (5,)),
    ((2, 3), (3, 2)),
    ((2, 3, 4), (3, 4, 2)),
    ((2, 3, 4), (3, -1, 2)),
    ((2, 3, 4), (3, -3, 2)),  # -3 is treated as a -1 and is valid.
]


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('a_shape,b_shape', _reshape_shape)
@chainer.testing.parameterize_pytest('shape_type', [tuple, list])
@chainer.testing.parameterize_pytest('contiguous', ['C', None])
@chainer.testing.parameterize_pytest('is_module', [True, False])
class TestReshape(op_utils.NumpyOpTest):

    def generate_inputs(self):
        a = array_utils.shaped_arange(self.a_shape, 'float64')
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        b_shape = self.b_shape
        shape_type = self.shape_type
        if self.is_module:
            b = xp.reshape(a, shape_type(b_shape))
        else:
            b = a.reshape(shape_type(b_shape))
        if xp is chainerx:
            copied = (
                a._debug_data_memory_address
                != b._debug_data_memory_address)
        else:
            copied = a.ctypes.data != b.ctypes.data
        if copied:
            if xp is chainerx:
                assert b.is_contiguous
            else:
                assert b.flags.c_contiguous

        return xp.asarray(copied), b


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('is_module', [True, False])
@chainer.testing.parameterize_pytest('a_shape,b_shape', _reshape_shape)
@chainer.testing.parameterize_pytest('contiguous', ['C', None])
class TestReshapeArg(op_utils.NumpyOpTest):

    forward_accept_errors = (TypeError, chainerx.ChainerxError)

    def setup(self):
        if self.is_module and len(self.b_shape) > 1:
            # Skipping tests where the 'order' argument is unintentionally
            # given a shape value, since numpy won't raise any errors in this
            # case which you might expect at first.
            raise unittest.SkipTest(
                'NumPy won\'t raise error for unintentional argument '
                'unpacking')

    def generate_inputs(self):
        a = array_utils.shaped_arange(self.a_shape, 'float64')
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        b_shape = self.b_shape
        if self.is_module:
            # TypeError/chainerx.ChainerxError in case b_shape is empty.
            b = xp.reshape(a, *b_shape)
        else:
            # TypeError/chainerx.ChainerxError in case b_shape is empty.
            b = a.reshape(*b_shape)

        if xp is chainerx:
            if self.contiguous == 'C':
                assert b.is_contiguous
                assert (a._debug_data_memory_address
                        == b._debug_data_memory_address), (
                            'Reshape must be done without copy')
        return b,


@pytest.mark.parametrize('shape1,shape2', [
    ((), (0,)),
    ((), (2,)),
    ((), (1, 2,)),
    ((0,), (1,)),
    ((0,), (1, 1, 1)),
    ((2, 3), (2, 3, 2)),
    ((2, 3, 4), (2, 3, 5)),
])
def test_reshape_invalid(shape1, shape2):
    def check(a_shape, b_shape):
        a = array_utils.create_dummy_ndarray(chainerx, a_shape, 'float32')
        with pytest.raises(chainerx.DimensionError):
            a.reshape(b_shape)

    check(shape1, shape2)
    check(shape2, shape1)


@pytest.mark.parametrize('shape1,shape2', [
    ((2, 3, 4), (5, -1, 3)),  # Not divisible.
    ((2, 3, 4), (-1, -1, 3)),  # More than one dimension cannot be inferred.
    ((2, 3, 4), (-2, 4, -1)),
])
def test_reshape_invalid_cannot_infer(shape1, shape2):
    a = array_utils.create_dummy_ndarray(chainerx, shape1, 'float32')
    with pytest.raises(chainerx.DimensionError):
        a.reshape(shape2)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('shape,axis', [
    ((), None),
    ((0,), None),
    ((1,), None),
    ((1, 1), None),
    ((1, 0, 1), None),
    ((3,), None),
    ((3, 1), None),
    ((1, 3), None),
    ((2, 0, 3), None),
    ((2, 4, 3), None),
    ((2, 1, 3), 1),
    ((2, 1, 3), -2),
    ((1, 2, 1, 3, 1, 1, 4), None),
    ((1, 2, 1, 3, 1, 1, 4), (2, 0, 4)),
    ((1, 2, 1, 3, 1, 1, 4), (-2, 0, 4)),
])
@chainer.testing.parameterize_pytest('is_module', [True, False])
class TestSqueeze(op_utils.NumpyOpTest):

    def generate_inputs(self):
        a = array_utils.shaped_arange(self.shape, 'float32')
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        axis = self.axis
        if self.is_module:
            b = xp.squeeze(a, axis)
        else:
            b = a.squeeze(axis)
        return b,


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(chainerx.DimensionError, ValueError))
@pytest.mark.parametrize('shape,axis', [
    ((2, 1, 3), 0),
    ((2, 1, 3), -1),
    ((2, 1, 3), (1, 2)),
    ((2, 1, 3), (1, -1)),
    ((2, 1, 3), (1, 1)),
])
def test_squeeze_invalid(is_module, xp, shape, axis):
    a = xp.ones(shape, 'float32')
    if is_module:
        return xp.squeeze(a, axis)
    else:
        return a.squeeze(axis)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('src_shape,dst_shape', [
    ((), ()),
    ((1,), (2,)),
    ((1, 1), (2, 2)),
    ((1, 1), (1, 2)),
    ((2,), (3, 2)),
])
class TestBroadcastTo(op_utils.NumpyOpTest):

    def generate_inputs(self):
        a = array_utils.shaped_arange(self.src_shape, 'float32')
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        b = xp.broadcast_to(a, self.dst_shape)
        return b,


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(chainerx.DimensionError, ValueError))
@pytest.mark.parametrize(('src_shape,dst_shape'), [
    ((3,), (2,)),
    ((3,), (3, 2)),
    ((1, 3), (3, 2)),
])
def test_broadcast_to_invalid(xp, src_shape, dst_shape):
    a = xp.ones(src_shape, 'float32')
    return xp.broadcast_to(a, dst_shape)


def _make_inputs(shapes, dtypes):
    # Generates input ndarrays.
    assert isinstance(shapes, (list, tuple))
    assert isinstance(dtypes, (list, tuple))
    assert len(shapes) == len(dtypes)

    inputs = []
    for i, (shape, dtype) in enumerate(zip(shapes, dtypes)):
        size = array_utils.total_size(shape)
        a = numpy.arange(i * 100, i * 100 + size)
        a = a.reshape(shape)
        a = a.astype(dtype)
        inputs.append(a)

    assert len(inputs) > 0
    return tuple(inputs)


class ConcatenateTestBase(op_utils.NumpyOpTest):

    chx_expected_dtype = None
    dtypes = None

    def setup(self):
        # Skip backward/double-backward tests for int dtypes
        if any(numpy.dtype(dt).kind != 'f' for dt in self.dtypes):
            self.skip_backward_test = True
            self.skip_double_backward_test = True

        # TODO(niboshi): Fix strides for 0-size inputs
        if any(0 in shape for shape in self.shapes):
            self.check_numpy_strides_compliance = False

        if any(dt == 'float16' for dt in self.dtypes):
            self.check_backward_options.update({'rtol': 1e-3, 'atol': 1e-3})

    def generate_inputs(self):
        return _make_inputs(self.shapes, self.dtypes)

    def forward_xp(self, inputs, xp):
        if self.axis is _unspecified:
            b = xp.concatenate(inputs)
        else:
            b = xp.concatenate(inputs, self.axis)

        if self.chx_expected_dtype is not None:
            b = dtype_utils.cast_if_numpy_array(xp, b, self.chx_expected_dtype)
        return b,


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('shapes,axis', [
    ([(0,)], 0),
    ([(1,)], 0),
    ([(0,), (0,)], 0),
    ([(0,), (1,)], 0),
    ([(1,), (1,)], 0),
    ([(0, 0,), (0, 0,)], 0),
    ([(0, 0,), (0, 0,)], 1),
    ([(1, 0,), (1, 0,)], 0),
    ([(1, 0,), (1, 0,)], 1),
    ([(1, 0,), (1, 0,)], 2),
    ([(3, 4, 5)], 0),
    ([(2, 3, 1), (2, 3, 1)], 1),
    ([(2, 3, 2), (2, 4, 2), (2, 3, 2)], 1),
    ([(2, 3, 2), (2, 4, 2), (3, 3, 2)], 1),
    ([(4, 10), (5, 10)], 0),
    ([(4, 10), (4, 8)], 0),
    ([(4, 4), (5,)], 0),
    ([(4, 4), (4,)], 0),
    ([(2, 3), (2, 3)], 10),
    ([(2, 3), (2, 3)], -1),
    ([(2, 3), (2, 3)], None),
    ([(2, 3), (4, 5)], None),
    ([(2, 3), (4, 5, 1)], None),
    ([(2, 3), (4, 5, 1), (4,)], None),
    ([(2, 3), (2, 3)], _unspecified),
    ([(2, 3), (4, 5)], _unspecified),
])
class TestConcatenate(ConcatenateTestBase):

    forward_accept_errors = (chainerx.DimensionError, ValueError)

    def setup(self):
        self.dtypes = ['float32'] * len(self.shapes)
        super().setup()


def test_concatenate_insufficient_inputs():
    with pytest.raises(chainerx.DimensionError):
        chainerx.concatenate([])


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('shapes,axis', [
    ([(0,), (0,)], 0),
    ([(0,), (1,)], 0),
    ([(1,), (1,)], 0),
    ([(0, 0,), (0, 0,)], 0),
    ([(0, 0,), (0, 0,)], 1),
    ([(1, 0,), (1, 0,)], 0),
    ([(1, 0,), (1, 0,)], 1),
    ([(2, 3, 1), (2, 3, 1)], 1),
    ([(4, 10), (5, 10)], 0),
    ([(2, 3), (2, 3)], None),
    ([(2, 3), (4, 5)], None),
    ([(2, 3), (4, 5, 1)], None),
    ([(2, 3), (2, 3)], _unspecified),
])
@chainer.testing.parameterize_pytest(
    'dtypes,chx_expected_dtype', dtype_utils.result_dtypes_two_arrays)
class TestConcatenateTwoArraysMixedDtypes(ConcatenateTestBase):
    pass


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('shapes,axis', [
    ([(0,), (0,), (0,)], 0),
    ([(0,), (1,), (1,)], 0),
    ([(2, 3, 2), (2, 4, 2), (2, 3, 2)], 1),
    ([(2, 3), (4, 5), (4, 2)], None),
    ([(2, 3), (4, 5, 1), (4,)], None),
    ([(2, 3), (2, 3), (1, 3)], _unspecified),
])
@chainer.testing.parameterize_pytest(
    'dtypes,chx_expected_dtype', dtype_utils.result_dtypes_three_arrays)
class TestConcatenateThreeArraysMixedDtypes(ConcatenateTestBase):
    pass


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('shapes,axis', [
    ([(0,)], -1),
    ([(0,)], 0),
    ([(0,)], 1),
    ([(0,)], 2),
    ([(1,)], -1),
    ([(1,)], 0),
    ([(1,)], 1),
    ([(1,)], 2),
    ([(0,), (0,)], 0),
    ([(0,), (0,)], 1),
    ([(0, 0,), (0, 0,)], 0),
    ([(0, 0,), (0, 0,)], 1),
    ([(1, 0,), (1, 0,)], 0),
    ([(1, 0,), (1, 0,)], 1),
    ([(1, 0,), (1, 0,)], 2),
    ([(3, 4, 5), (3, 4, 5), (3, 4, 5)], None),
    ([(3, 4, 5), (3, 4, 5), (3, 4, 5)], 0),
    ([(3, 4, 5), (3, 4, 5), (3, 4, 5)], 1),
    ([(3, 4, 5), (3, 4, 5), (3, 4, 5)], 2),
    ([(3, 4, 5), (3, 4, 5), (3, 4, 5)], 3),
    ([(3, 4, 5), (3, 4, 5), (3, 4, 5)], 4),
    ([(2, 3, 2), (2, 4, 2), (2, 3, 2)], 1),
])
class TestStack(op_utils.NumpyOpTest):

    forward_accept_errors = (chainerx.DimensionError, ValueError)

    def generate_inputs(self):
        return _make_inputs(self.shapes, ['float32'] * len(self.shapes))

    def forward_xp(self, inputs, xp):
        axis = self.axis
        if axis is None:
            b = xp.stack(inputs)
        else:
            b = xp.stack(inputs, axis)
        return b,


def test_stack_insufficient_inputs():
    with pytest.raises(chainerx.DimensionError):
        chainerx.stack([])
    with pytest.raises(chainerx.DimensionError):
        chainerx.stack([], 0)


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('shape,indices_or_sections,axis', [
    ((2,), 1, 0),
    ((2,), [], 0),
    ((2,), [1, 2], 0),
    ((2,), [-5, -3], 0),
    ((2, 4), 1, 0),
    ((2, 4), 2, 1),
    ((2, 4), 2, -1),
    ((2, 4, 6), [], 0),
    ((2, 4, 6), [2, 4], 2),
    ((2, 4, 6), [2, -3], 2),
    ((2, 4, 6), [2, 8], 2),
    ((2, 4, 6), [4, 2], 2),
    ((2, 4, 6), [1, 3], -2),
    ((6,), numpy.array([1, 2]), 0),  # indices with 1-d numpy array
    ((6,), numpy.array([2]), 0),  # indices with (1,)-shape numpy array
    ((6,), numpy.array(2), 0),  # sections numpy scalar
    ((6,), numpy.array(2.0), 0),  # sections with numpy scalar, float
    ((6,), 2.0, 0),  # float type sections, without fraction
    # indices with empty numpy indices
    ((6,), numpy.array([], numpy.int32), 0),
    ((6,), numpy.array([], numpy.float64), 0),
])
class TestSplit(op_utils.NumpyOpTest):

    def setup(self):
        # TODO(niboshi): There's a bug in backward of split() in which the
        # gradient shape differs from the input if indices are not in the
        # sorted order. Fix this.
        indices_or_sections = self.indices_or_sections
        if (isinstance(indices_or_sections, list) and
                sorted(indices_or_sections) != indices_or_sections):
            self.skip_backward_test = True
            self.skip_double_backward_test = True

    def generate_inputs(self):
        a = array_utils.create_dummy_ndarray(numpy, self.shape, 'float32')
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        b = xp.split(a, self.indices_or_sections, self.axis)
        assert isinstance(b, list)
        return tuple(b)


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(
        chainerx.DimensionError, IndexError, ValueError, TypeError,
        ZeroDivisionError))
@pytest.mark.parametrize('shape,indices_or_sections,axis', [
    ((), 1, 0),
    ((2,), 3, 0),
    ((2, 4), 0, 0),
    ((2, 4), -1, 1),
    ((2, 4), 1, 2),  # Axis out of range.
    ((2, 4), 3, 1),  # Uneven split.
    ((6,), [2.0], 0),  # float type indices
    ((6,), 2.1, 0),  # float type sections, with fraction
    # indices with (1,)-shape numpy array, float
    ((6,), numpy.array([2.0]), 0),
    # sections with numpy scalar, float with fraction
    ((6,), numpy.array(2.1), 0),
    ((2,), [1, 2.0], 0),  # indices with mixed type
    ((6,), '2', 0),  # Invalid type
    # indices with empty numpy indices
    ((6,), numpy.array([[], []], numpy.int32), 0),
    ((6,), numpy.array([[], []], numpy.float64), 0),
])
def test_split_invalid(xp, shape, indices_or_sections, axis):
    a = array_utils.create_dummy_ndarray(xp, shape, 'float32')
    return xp.split(a, indices_or_sections, axis)


@op_utils.op_test(['native:0'])
@chainer.testing.parameterize_pytest('shape,axis1,axis2', [
    ((1, 1), 0, 1),
    ((2, 4), -1, 1),
    ((1, 2, 2), 0, 1),
    ((1, 2, 3, 4), 0, 2),
    ((3, 2, 1, 2, 3), 0, 4),
    ((1, 2, 4, 3, 1), 0, -2),
    ((1, 2, 4, 2, 1), 0, 0),
    ((1, 3, 3, 1), -1, -4),
])
@chainer.testing.parameterize_pytest('is_module', [True, False])
class TestSwapaxes(op_utils.NumpyOpTest):

    def setup(self, dtype):
        # Skip backward/double-backward tests for int dtypes
        if numpy.dtype(dtype).kind != 'f':
            self.skip_backward_test = True
            self.skip_double_backward_test = True
        self.dtype = dtype

    def generate_inputs(self):
        a = array_utils.create_dummy_ndarray(numpy, self.shape, self.dtype)
        return a,

    def forward_xp(self, inputs, xp):
        a, = inputs
        if self.is_module:
            b = xp.swapaxes(a, self.axis1, self.axis2)
        else:
            b = a.swapaxes(self.axis1, self.axis2)
        return b,


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(
        chainerx.DimensionError, numpy.AxisError))
@pytest.mark.parametrize('shape,axis1,axis2', [
    # Axis out of range.
    ((), 1, 0),
    ((2,), 3, 0),
    ((2, 4), 1, 2),
    ((1, 1, 2), -1, -4)
])
def test_swap_invalid(xp, shape, axis1, axis2):
    a = array_utils.create_dummy_ndarray(xp, shape, 'float32')
    return xp.swapaxes(a, axis1, axis2)
