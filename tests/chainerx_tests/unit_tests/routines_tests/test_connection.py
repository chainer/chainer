import unittest

import chainer
from chainer import functions as F
import numpy
import pytest

import chainerx

from chainerx_tests import array_utils
from chainerx_tests import dtype_utils
from chainerx_tests import op_utils


# A special parameter object used to represent an unspecified argument.
class Unspecified(object):
    pass


def _create_conv_args(
        xp, device, x_shape, w_shape, b_shape, stride, pad, cover_all,
        float_dtype):
    x = array_utils.create_dummy_ndarray(xp, x_shape, float_dtype)
    w = array_utils.create_dummy_ndarray(xp, w_shape, float_dtype)
    if b_shape is None:
        b = None
    else:
        b = array_utils.create_dummy_ndarray(xp, b_shape, float_dtype)
    if device.backend.name == 'cuda':  # cover_all is not supported by CUDA.
        cover_all = False
    return x, w, b, stride, pad, cover_all


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # without bias
    chainer.testing.product([
        chainer.testing.from_pytest_parameterize(
            'x_shape,w_shape,b_shape,stride,pad', [
                ((1, 3), (5, 3), None, 1, 0),
                ((1, 3, 4), (5, 3, 2), None, 3, 2),
                ((2, 3, 4, 4), (2, 3, 3, 3), None, 2, (2, 0)),
                ((2, 3, 2, 6, 3), (2, 3, 1, 3, 2), None, (1, 2, 3), (2, 0, 1)),
            ]),
        chainer.testing.from_pytest_parameterize(
            'in_dtypes,out_dtype', dtype_utils.result_dtypes_two_arrays)
    ]) +
    # with bias
    chainer.testing.product([
        chainer.testing.from_pytest_parameterize(
            'x_shape,w_shape,b_shape,stride,pad', [
                ((1, 3), (5, 3), (5,), 1, 0),
                ((2, 3, 4), (5, 3, 1), (5,), 1, 0),
                ((1, 3, 4), (5, 3, 2), (5,), 3, 2),
                ((2, 3, 4, 4), (2, 3, 3, 3), (2,), 1, 0),
                ((1, 3, 4, 4), (2, 3, 3, 3), (2,), (1, 2), 1),
                ((1, 3, 4, 4), (2, 3, 3, 3), (2,), 2, (2, 0)),
                ((1, 3, 2, 6, 3), (2, 3, 1, 3, 2), (2,), 2, (2, 0, 1)),
                ((1, 3, 2, 6, 3), (2, 3, 1, 3, 2), (2,), (1, 2, 3), (2, 0, 1)),
            ]),
        chainer.testing.from_pytest_parameterize(
            'in_dtypes,out_dtype', dtype_utils.result_dtypes_three_arrays)
    ])
))
@chainer.testing.parameterize_pytest('cover_all', [True, False])
class TestConv(op_utils.ChainerOpTest):

    def setup(self):
        if len(self.in_dtypes) == 3:
            x_dtype, w_dtype, b_dtype = self.in_dtypes
        else:
            (x_dtype, w_dtype), b_dtype = self.in_dtypes, None

        x_kind = numpy.dtype(x_dtype).kind
        w_kind = numpy.dtype(w_dtype).kind
        b_kind = None if b_dtype is None else numpy.dtype(b_dtype).kind

        device = chainerx.get_default_device()
        if device.backend.name == 'cuda' and len(self.x_shape) <= 3:
            # TODO(hvy): Support 1 dimensional convolution with CUDA.
            pytest.skip('cudnn does not support 1-dim convolution')
        if device.backend.name == 'cuda' and self.cover_all:
            pytest.skip('cudnn does not support cover_all')

        # Skip backward/double-backward tests for int dtypes
        if (x_kind != 'f' and w_kind != 'f'
                and (b_kind is None or b_kind != 'f')):
            self.skip_backward_test = True
            self.skip_double_backward_test = True

        if (x_dtype == 'float16' or w_dtype == 'float16'
                or b_dtype == 'float16'):
            self.check_forward_options.update({'rtol': 5e-2, 'atol': 5e-3})
            self.check_backward_options.update({
                'eps': 2 ** -3, 'rtol': 1e-1, 'atol': 1e-2})
        else:
            self.check_forward_options.update({'rtol': 1e-3})
            self.check_backward_options.update({
                'eps': 1e-2, 'rtol': 1e-3, 'atol': 1e-4})
        self.check_double_backward_options.update({
            'rtol': 5e-2, 'atol': 5e-3})

    def generate_inputs(self):
        x_shape = self.x_shape
        w_shape = self.w_shape
        b_shape = self.b_shape
        if len(self.in_dtypes) == 3:
            x_dtype, w_dtype, b_dtype = self.in_dtypes
        else:
            (x_dtype, w_dtype), b_dtype = self.in_dtypes, None
        x = array_utils.uniform(x_shape, x_dtype)
        w = array_utils.uniform(w_shape, w_dtype)
        if b_shape is None:
            return x, w
        else:
            b = array_utils.uniform(b_shape, b_dtype)
            return x, w, b

    def forward_chainerx(self, inputs):
        if len(inputs) == 2:
            (x, w), b = inputs, None
        else:
            x, w, b = inputs
        y = chainerx.conv(x, w, b, self.stride, self.pad, self.cover_all)
        return y,

    def forward_chainer(self, inputs):
        if len(inputs) == 2:
            (x, w), b = inputs, None
        else:
            x, w, b = inputs
        if x.dtype.kind != 'f':
            x = F.cast(x, 'float64')
        if w.dtype.kind != 'f':
            w = F.cast(w, 'float64')
        if b is not None and b.dtype.kind != 'f':
            b = F.cast(b, 'float64')
        y = F.convolution_nd(
            x, w, b, self.stride, self.pad, self.cover_all)
        y = F.cast(y, self.out_dtype)
        return y,


@pytest.mark.parametrize('x_shape,w_shape,b_shape,stride,pad', [
    # Mismatched x and w input channels.
    ((1, 3, 4, 3), (5, 4, 2, 2), (5,), 3, 2),
    # Mismatched x and w dimensions.
    ((2, 3, 4, 3), (5, 3, 2, 2, 1), (5,), 3, 2),
    ((1, 3, 4, 3), (5, 3, 2, 2), (6,), 1, 0),  # Mismatched w and b.
    ((2, 3, 4, 3), (5, 3, 2, 2), None, (1,), 0),  # Wrong number of strides.
    ((1, 3, 4, 3), (5, 3, 2, 2), None, 3, (2,)),  # Wrong number of paddings.
])
@pytest.mark.parametrize('cover_all', [True, False])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_conv_invalid(
        device, x_shape, w_shape, b_shape, stride, pad, cover_all,
        float_dtype):
    with pytest.raises(chainerx.DimensionError):
        chainerx.conv(
            *_create_conv_args(
                chainerx, device, x_shape, w_shape, b_shape, stride, pad,
                cover_all, float_dtype))


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    # without bias
    chainer.testing.product([
        chainer.testing.from_pytest_parameterize(
            'x_shape,w_shape,b_shape,stride,pad', [
                ((1, 3), (3, 5), None, 1, 0),
                ((1, 3, 4), (3, 5, 2), None, 3, 2),
                ((2, 3, 4, 4), (3, 2, 3, 3), None, 2, (2, 0)),
                ((2, 3, 5, 6, 3), (3, 2, 1, 3, 2), None, (1, 2, 3), (2, 0, 1)),
            ]),
        chainer.testing.from_pytest_parameterize(
            'in_dtypes,out_dtype', dtype_utils.result_dtypes_two_arrays)
    ]) +
    # with bias
    chainer.testing.product([
        chainer.testing.from_pytest_parameterize(
            'x_shape,w_shape,b_shape,stride,pad', [
                ((1, 3), (3, 5), (5,), 1, 0),
                ((2, 3, 4), (3, 5, 1), (5,), 1, 0),
                ((1, 3, 4), (3, 5, 2), (5,), 3, 2),
                ((2, 3, 4, 4), (3, 2, 3, 3), (2,), 1, 0),
                ((1, 3, 4, 4), (3, 2, 3, 3), (2,), (1, 2), 1),
                ((1, 3, 4, 4), (3, 2, 3, 3), (2,), 2, (2, 0)),
                ((1, 3, 5, 6, 3), (3, 2, 1, 3, 2), (2,), 2, (2, 0, 1)),
                ((1, 3, 5, 6, 3), (3, 2, 1, 3, 2), (2,), (1, 2, 3), (2, 0, 1)),
            ]),
        chainer.testing.from_pytest_parameterize(
            'in_dtypes,out_dtype', dtype_utils.result_dtypes_three_arrays)
    ])
))
# If None, outsize argument will be None.
@chainer.testing.parameterize_pytest('cover_all', [None, True, False])
class TestConvTranspose(op_utils.ChainerOpTest):

    def setup(self):
        if len(self.in_dtypes) == 3:
            x_dtype, w_dtype, b_dtype = self.in_dtypes
        else:
            (x_dtype, w_dtype), b_dtype = self.in_dtypes, None

        x_kind = numpy.dtype(x_dtype).kind
        w_kind = numpy.dtype(w_dtype).kind
        b_kind = None if b_dtype is None else numpy.dtype(b_dtype).kind

        device = chainerx.get_default_device()
        if device.backend.name == 'cuda' and len(self.x_shape) <= 3:
            # TODO(sonots): Support 1 dimensional convolution with CUDA.
            pytest.skip(
                'cuDNN does not support 1 dimensional convolution and throws '
                'DimensionError')
        if device.backend.name == 'cuda' and self.cover_all is True:
            pytest.skip(
                'outsize (for cover_all=True) is not supported by CUDA')

        # Skip backward/double-backward tests for int dtypes
        if (x_kind != 'f' and w_kind != 'f'
                and (b_kind is None or b_kind != 'f')):
            self.skip_backward_test = True
            self.skip_double_backward_test = True

        if (x_dtype == 'float16' or w_dtype == 'float16'
                or b_dtype == 'float16'):
            self.check_forward_options.update({'rtol': 5e-2, 'atol': 2e-2})
            self.check_backward_options.update({
                'eps': 2 ** -3, 'rtol': 1e-1, 'atol': 1e-2})
        else:
            self.check_forward_options.update({'rtol': 1e-3})
            self.check_backward_options.update({
                'eps': 1e-2, 'rtol': 1e-3, 'atol': 1e-4})
        self.check_double_backward_options.update({
            'rtol': 5e-2, 'atol': 5e-3})

        # Determine outsize
        cover_all = self.cover_all
        if cover_all is None:
            outsize = None
        else:
            x_shape = self.x_shape
            w_shape = self.w_shape
            stride = self.stride
            pad = self.pad
            in_dims = x_shape[2:]
            kernel_size = w_shape[2:]
            ndim = len(in_dims)
            stride_tup = (
                (stride,) * ndim if isinstance(stride, int) else stride)
            pad_tup = (pad,) * ndim if isinstance(pad, int) else pad
            outsize = tuple(
                chainer.utils.conv.get_deconv_outsize(d, k, s, p, cover_all)
                for (d, k, s, p)
                in zip(in_dims, kernel_size, stride_tup, pad_tup))
        self.outsize = outsize

    def generate_inputs(self):
        x_shape = self.x_shape
        w_shape = self.w_shape
        b_shape = self.b_shape
        if len(self.in_dtypes) == 3:
            x_dtype, w_dtype, b_dtype = self.in_dtypes
        else:
            (x_dtype, w_dtype), b_dtype = self.in_dtypes, None
        x = array_utils.uniform(x_shape, x_dtype)
        w = array_utils.uniform(w_shape, w_dtype)
        if b_shape is None:
            return x, w
        else:
            b = array_utils.uniform(b_shape, b_dtype)
            return x, w, b

    def forward_chainerx(self, inputs):
        if len(inputs) == 3:
            x, w, b = inputs
        else:
            (x, w), b = inputs, None
        y = chainerx.conv_transpose(
            x, w, b, self.stride, self.pad, self.outsize)
        return y,

    def forward_chainer(self, inputs):
        if len(inputs) == 3:
            x, w, b = inputs
        else:
            (x, w), b = inputs, None
        if x.dtype.kind != 'f':
            x = F.cast(x, 'float64')
        if w.dtype.kind != 'f':
            w = F.cast(w, 'float64')
        if b is not None and b.dtype.kind != 'f':
            b = F.cast(b, 'float64')
        y = chainer.functions.deconvolution_nd(
            x, w, b, self.stride, self.pad, self.outsize)
        y = F.cast(y, self.out_dtype)
        return y,


@pytest.mark.parametrize('x_shape,w_shape,b_shape,stride,pad,outsize', [
    # Mismatched x and w input channels.
    ((1, 3, 4, 3), (5, 4, 2, 2), (5,), 3, 2, None),
    # Mismatched x and w dimensions.
    ((2, 3, 4, 3), (3, 5, 2, 2, 1), (5,), 3, 2, None),
    ((1, 3, 4, 3), (3, 5, 2, 2), (6,), 1, 0, None),  # Mismatched w and b.
    # Wrong number of strides.
    ((2, 3, 4, 3), (3, 5, 2, 2), None, (1,), 0, None),
    # Wrong number of paddings.
    ((1, 3, 4, 3), (3, 5, 2, 2), None, 3, (2,), None),
    ((1, 3, 2, 6, 3), (3, 2, 1, 3, 2), (2,), 2, (2, 0, 1),
     (-1, 13, 4)),  # All output sizes must be non-negative
    # All output sizes must be non-negative
    ((1, 3, 2, 6, 3), (3, 2, 1, 3, 2), (2,), 2, (2, 0, 1), None),
    ((2, 3, 4), (3, 5, 1), (5,), 1, 0, (5,)),  # Output dims are inconsistent
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_conv_transpose_invalid(
        device, x_shape, w_shape, b_shape, stride, pad, outsize, float_dtype):
    dtype = float_dtype
    x = array_utils.create_dummy_ndarray(chainerx, x_shape, dtype)
    w = array_utils.create_dummy_ndarray(chainerx, w_shape, dtype)
    if b_shape is None:
        b = None
    else:
        b = array_utils.create_dummy_ndarray(chainerx, b_shape, float_dtype)

    with pytest.raises(chainerx.DimensionError):
        chainerx.conv_transpose(x, w, b, stride, pad, outsize)


@op_utils.op_test(['native:0', 'cuda:0'])
# TODO(imanishi): Add test cases for more than 2 ndim
@chainer.testing.parameterize(*(
    # without bias
    chainer.testing.product([
        chainer.testing.from_pytest_parameterize(
            'x_shape,w_shape,b_shape,n_batch_axes', [
                ((2, 3), (4, 3), None, Unspecified),
                ((5, 2, 3), (4, 3), None, 2),
                ((2, 3), (4, 3), Unspecified, Unspecified),
                ((5, 2, 3), (4, 6), Unspecified, Unspecified),
            ]),
        chainer.testing.from_pytest_parameterize(
            'in_dtypes,out_dtype', dtype_utils.result_dtypes_two_arrays)
    ]) +
    # with bias
    chainer.testing.product([
        chainer.testing.from_pytest_parameterize(
            'x_shape,w_shape,b_shape,n_batch_axes', [
                ((2, 3), (4, 3), (4,), Unspecified),
                ((2, 0), (3, 0), (3,), Unspecified),
                ((0, 2), (0, 2), (0,), Unspecified),
                ((0, 0), (0, 0), (0,), Unspecified),
                ((5, 2, 3), (4, 3), (4,), 2),
            ]),
        chainer.testing.from_pytest_parameterize(
            'in_dtypes,out_dtype', dtype_utils.result_dtypes_three_arrays)
    ])
))
class TestLinear(op_utils.OpTest):

    def setup(self):
        if len(self.in_dtypes) == 3:
            x_dtype, w_dtype, b_dtype = self.in_dtypes
        else:
            (x_dtype, w_dtype), b_dtype = self.in_dtypes, None

        x_kind = numpy.dtype(x_dtype).kind
        w_kind = numpy.dtype(w_dtype).kind
        b_kind = None if b_dtype is None else numpy.dtype(b_dtype).kind

        device = chainerx.get_default_device()
        if device.backend.name == 'cuda' and (
                x_kind != 'f' or w_kind != 'f' or b_kind != 'f'):
            raise unittest.SkipTest('CUDA dot does not support integers.')

        # Skip backward/double-backward tests for int dtypes
        if (x_kind != 'f' and w_kind != 'f'
                and (b_kind is None or b_kind != 'f')):
            self.skip_backward_test = True
            self.skip_double_backward_test = True

        # Skip backward/double-backward tests if the output will be
        # disconnected.
        # TODO(niboshi): Remove this skip condition after enabling backward()
        # for such cases.
        if 0 in self.x_shape or 0 in self.w_shape:
            self.skip_backward_test = True
            self.skip_double_backward_test = True

        if (x_dtype == 'float16' or w_dtype == 'float16'
                or b_dtype == 'float16'):
            self.check_forward_options.update({
                'rtol': 1e-2, 'atol': 1e-2})
            self.check_backward_options.update({
                'rtol': 1e-2, 'atol': 1e-2})
            self.check_double_backward_options.update({
                'rtol': 1e-2, 'atol': 1e-2})

    def generate_inputs(self):
        x_shape = self.x_shape
        w_shape = self.w_shape
        b_shape = self.b_shape
        if len(self.in_dtypes) == 3:
            x_dtype, w_dtype, b_dtype = self.in_dtypes
        else:
            (x_dtype, w_dtype), b_dtype = self.in_dtypes, None
        x = array_utils.uniform(x_shape, x_dtype)
        w = array_utils.uniform(w_shape, w_dtype)
        if b_shape in (None, Unspecified):
            return x, w
        else:
            b = array_utils.uniform(b_shape, b_dtype)
            return x, w, b

    def forward_chainerx(self, inputs):
        if len(inputs) == 3:
            x, w, b = inputs
        else:
            (x, w), b = inputs, None

        n_batch_axes = self.n_batch_axes

        if b is Unspecified:
            y = chainerx.linear(x, w)
        elif n_batch_axes is Unspecified:
            y = chainerx.linear(x, w, b)
        else:
            y = chainerx.linear(x, w, b, n_batch_axes)
        return y,

    def forward_expected(self, inputs):
        if len(inputs) == 3:
            x, w, b = inputs
        else:
            (x, w), b = inputs, None

        n_batch_axes = self.n_batch_axes
        x_shape = self.x_shape
        w_shape = self.w_shape
        out_dtype = self.out_dtype

        if n_batch_axes is Unspecified:
            n_batch_axes = 1
        y_shape = x_shape[:n_batch_axes] + (w_shape[0],)
        x_ = x.reshape(numpy.prod(x_shape[:n_batch_axes]),
                       numpy.prod(x_shape[n_batch_axes:]))
        x_ = x_.astype(out_dtype)
        w_ = w.astype(out_dtype)
        y = x_.dot(w_.T).reshape(y_shape)
        if b is not None:
            y += b

        assert y.dtype == out_dtype
        return y,
