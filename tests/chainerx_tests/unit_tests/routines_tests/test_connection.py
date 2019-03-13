import chainer
import numpy
import pytest

import chainerx

from chainerx_tests import array_utils
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
@chainer.testing.parameterize_pytest('x_shape,w_shape,b_shape,stride,pad', [
    ((1, 3), (5, 3), (5,), 1, 0),
    ((1, 3), (5, 3), None, 1, 0),
    ((2, 3, 4), (5, 3, 1), (5,), 1, 0),
    ((1, 3, 4), (5, 3, 2), (5,), 3, 2),
    ((1, 3, 4), (5, 3, 2), None, 3, 2),
    ((2, 3, 4, 4), (2, 3, 3, 3), (2,), 1, 0),
    ((1, 3, 4, 4), (2, 3, 3, 3), (2,), (1, 2), 1),
    ((1, 3, 4, 4), (2, 3, 3, 3), (2,), 2, (2, 0)),
    ((2, 3, 4, 4), (2, 3, 3, 3), None, 2, (2, 0)),
    ((1, 3, 2, 6, 3), (2, 3, 1, 3, 2), (2,), 2, (2, 0, 1)),
    ((1, 3, 2, 6, 3), (2, 3, 1, 3, 2), (2,), (1, 2, 3), (2, 0, 1)),
    ((2, 3, 2, 6, 3), (2, 3, 1, 3, 2), None, (1, 2, 3), (2, 0, 1)),
])
@chainer.testing.parameterize_pytest('cover_all', [True, False])
class TestConv(op_utils.ChainerOpTest):

    def setup(self, float_dtype):

        device = chainerx.get_default_device()
        if device.backend.name == 'cuda' and len(self.x_shape) <= 3:
            # TODO(hvy): Support 1 dimensional convolution with CUDA.
            pytest.skip('cudnn does not support 1-dim convolution')
        if device.backend.name == 'cuda' and self.cover_all:
            pytest.skip('cudnn does not support cover_all')
        if device.backend.name == 'native' and float_dtype == 'float16':
            # TODO(niboshi): Fix accuracy
            pytest.skip('Native float16 operation has insufficient accuracy')

        self.dtype = float_dtype

        if float_dtype == 'float16':
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
        dtype = self.dtype
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        w = numpy.random.uniform(-1, 1, w_shape).astype(dtype)
        if b_shape is None:
            return x, w
        else:
            b = numpy.random.uniform(-1, 1, b_shape).astype(dtype)
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
        y = chainer.functions.convolution_nd(
            x, w, b, self.stride, self.pad, self.cover_all)
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
@chainer.testing.parameterize_pytest('x_shape,w_shape,b_shape,stride,pad', [
    ((1, 3), (3, 5), (5,), 1, 0),
    ((1, 3), (3, 5), None, 1, 0),
    ((2, 3, 4), (3, 5, 1), (5,), 1, 0),
    ((1, 3, 4), (3, 5, 2), (5,), 3, 2),
    ((1, 3, 4), (3, 5, 2), None, 3, 2),
    ((2, 3, 4, 4), (3, 2, 3, 3), (2,), 1, 0),
    ((1, 3, 4, 4), (3, 2, 3, 3), (2,), (1, 2), 1),
    ((1, 3, 4, 4), (3, 2, 3, 3), (2,), 2, (2, 0)),
    ((2, 3, 4, 4), (3, 2, 3, 3), None, 2, (2, 0)),
    ((1, 3, 5, 6, 3), (3, 2, 1, 3, 2), (2,), 2, (2, 0, 1)),
    ((1, 3, 5, 6, 3), (3, 2, 1, 3, 2), (2,), (1, 2, 3), (2, 0, 1)),
    ((2, 3, 5, 6, 3), (3, 2, 1, 3, 2), None, (1, 2, 3), (2, 0, 1)),
])
# If None, outsize argument will be None.
@chainer.testing.parameterize_pytest('cover_all', [None, True, False])
class TestConvTranspose(op_utils.ChainerOpTest):

    def setup(self, float_dtype):
        self.dtype = float_dtype

        device = chainerx.get_default_device()
        if device.backend.name == 'cuda' and len(self.x_shape) <= 3:
            # TODO(sonots): Support 1 dimensional convolution with CUDA.
            pytest.skip(
                'cuDNN does not support 1 dimensional convolution and throws '
                'DimensionError')
        if device.backend.name == 'cuda' and self.cover_all is True:
            pytest.skip(
                'outsize (for cover_all=True) is not supported by CUDA')

        if float_dtype == 'float16':
            self.check_forward_options.update({'rtol': 5e-2, 'atol': 5e-3})
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
        dtype = self.dtype
        x_shape = self.x_shape
        w_shape = self.w_shape
        b_shape = self.b_shape
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        w = numpy.random.uniform(-1, 1, w_shape).astype(dtype)
        if b_shape is None:
            return x, w
        else:
            b = numpy.random.uniform(-1, 1, b_shape).astype(dtype)
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
        y = chainer.functions.deconvolution_nd(
            x, w, b, self.stride, self.pad, self.outsize)
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
@chainer.testing.parameterize_pytest('x_shape,w_shape,b_shape,n_batch_axes', [
    ((2, 3), (4, 3), (4,), Unspecified),
    ((2, 0), (3, 0), (3,), Unspecified),
    ((0, 2), (0, 2), (0,), Unspecified),
    ((0, 0), (0, 0), (0,), Unspecified),
    ((2, 3), (4, 3), Unspecified, Unspecified),
    ((2, 3), (4, 3), None, Unspecified),
    ((5, 2, 3), (4, 6), Unspecified, Unspecified),
    ((5, 2, 3), (4, 3), None, 2),
    ((5, 2, 3), (4, 3), (4,), 2),
    # TODO(imanishi): Add test cases for more than 2 ndim
])
class TestLinear(op_utils.OpTest):

    def setup(self, dtype):
        device = chainerx.get_default_device()
        # TODO(imanishi): Remove the skip after supporting non-float dot on
        # CUDA
        if device.name == 'cuda:0' and numpy.dtype(dtype).kind != 'f':
            pytest.skip('non-float dot is not supported')

        # Skip backward/double-backward tests for int dtypes
        if numpy.dtype(dtype).kind != 'f':
            self.skip_backward_test = True
            self.skip_double_backward_test = True

        if dtype == 'float16':
            self.check_forward_options.update({
                'rtol': 1e-2, 'atol': 1e-2})
            self.check_backward_options.update({
                'rtol': 1e-2, 'atol': 1e-2})
            self.check_double_backward_options.update({
                'rtol': 1e-2, 'atol': 1e-2})

        self.dtype = dtype

    def generate_inputs(self):
        x_shape = self.x_shape
        w_shape = self.w_shape
        b_shape = self.b_shape
        dtype = self.dtype
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        w = numpy.random.uniform(-1, 1, w_shape).astype(dtype)
        if b_shape in (None, Unspecified):
            return x, w
        else:
            b = numpy.random.uniform(-1, 1, b_shape).astype(dtype)
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

        if n_batch_axes is Unspecified:
            n_batch_axes = 1
        y_shape = x_shape[:n_batch_axes] + (w_shape[0],)
        x = x.reshape(numpy.prod(x_shape[:n_batch_axes]),
                      numpy.prod(x_shape[n_batch_axes:]))
        y = x.dot(w.T).reshape(y_shape)
        if b is not None:
            y += b

        return y,
