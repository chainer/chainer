import chainer
import numpy
import pytest

import chainerx

from chainerx_tests import array_utils


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


@pytest.mark.parametrize('x_shape,w_shape,b_shape,stride,pad', [
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
@pytest.mark.parametrize('cover_all', [True, False])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_conv(
        device, x_shape, w_shape, b_shape, stride, pad, cover_all,
        float_dtype):
    if device.backend.name == 'cuda' and len(x_shape) <= 3:
        # cuDNN does not support 1 dimensional convolution and throws
        # DimensionError.
        # TODO(hvy): Support 1 dimensional convolution with CUDA.
        return chainerx.testing.ignore()

    def create_args(xp):
        return _create_conv_args(
            xp, device, x_shape, w_shape, b_shape, stride, pad, cover_all,
            float_dtype)
    chainerx.testing.assert_allclose_ex(
        chainerx.conv(*create_args(chainerx)),
        chainer.functions.convolution_nd(*create_args(numpy)).data,
        float16_rtol=3e-3, float16_atol=3e-3, strides_check=False)


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


def _get_conv_transpose_outsize(x_shape, w_shape, stride, pad, cover_all):
    in_dims = x_shape[2:]
    kernel_size = w_shape[2:]
    ndim = len(in_dims)
    stride_tup = (stride,) * ndim if isinstance(stride, int) else stride
    pad_tup = (pad,) * ndim if isinstance(pad, int) else pad
    return tuple(chainer.utils.conv.get_deconv_outsize(d, k, s, p, cover_all)
                 for (d, k, s, p)
                 in zip(in_dims, kernel_size, stride_tup, pad_tup))


def _create_conv_transpose_args(
        xp, device, x_shape, w_shape, b_shape, stride, pad, outsize,
        float_dtype):
    x = array_utils.create_dummy_ndarray(xp, x_shape, float_dtype)
    w = array_utils.create_dummy_ndarray(xp, w_shape, float_dtype)
    if b_shape is None:
        b = None
    else:
        b = array_utils.create_dummy_ndarray(xp, b_shape, float_dtype)
    return x, w, b, stride, pad, outsize


@pytest.mark.parametrize('x_shape,w_shape,b_shape,stride,pad', [
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
@pytest.mark.parametrize('cover_all', [None, True, False])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_conv_transpose(
        device, x_shape, w_shape, b_shape, stride, pad, cover_all,
        float_dtype):
    if device.backend.name == 'cuda' and len(x_shape) <= 3:
        # cuDNN does not support 1 dimensional convolution and throws
        # DimensionError.
        # TODO(sonots): Support 1 dimensional convolution with CUDA.
        return chainerx.testing.ignore()
    if device.backend.name == 'cuda' and cover_all is True:
        # outsize (for cover_all=True) is not supported by CUDA.
        return chainerx.testing.ignore()

    def create_args(xp):
        if cover_all is None:
            outsize = None
        else:
            outsize = _get_conv_transpose_outsize(
                x_shape, w_shape, stride, pad, cover_all)
        return _create_conv_transpose_args(
            xp, device, x_shape, w_shape, b_shape, stride, pad, outsize,
            float_dtype)

    chainerx.testing.assert_allclose_ex(
        chainerx.conv_transpose(
            *create_args(chainerx)),
        chainer.functions.deconvolution_nd(*create_args(numpy)).data,
        rtol=1e-3, float16_rtol=1e-2, float16_atol=1e-2, strides_check=False)


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
    with pytest.raises(chainerx.DimensionError):
        chainerx.conv_transpose(
            *_create_conv_transpose_args(
                chainerx, device, x_shape, w_shape, b_shape, stride, pad,
                outsize, float_dtype))


@pytest.mark.parametrize('x_shape,w_shape,b_shape,n_batch_axes', [
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
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_linear(device, x_shape, w_shape, b_shape, n_batch_axes, dtype):
    # TODO(imanishi): Remove the skip after supporting non-float dot on CUDA
    if device.name == 'cuda:0' and numpy.dtype(dtype).kind != 'f':
        return chainerx.testing.ignore()
    x = array_utils.create_dummy_ndarray(numpy, x_shape, dtype)
    w = array_utils.create_dummy_ndarray(numpy, w_shape, dtype)
    b = (None if b_shape in (None, Unspecified)
         else array_utils.create_dummy_ndarray(numpy, b_shape, dtype))

    # Calculate chainerx_out
    chainerx_x = chainerx.array(x)
    chainerx_w = chainerx.array(w)
    chainerx_b = chainerx.array(b) if b is not None else None
    if b_shape is Unspecified:
        chainerx_out = chainerx.linear(chainerx_x, chainerx_w)
    elif n_batch_axes is Unspecified:
        chainerx_out = chainerx.linear(chainerx_x, chainerx_w, chainerx_b)
    else:
        chainerx_out = chainerx.linear(chainerx_x, chainerx_w, chainerx_b,
                                       n_batch_axes)

    # Calculate numpy_out
    if n_batch_axes is Unspecified:
        n_batch_axes = 1
    out_shape = x_shape[:n_batch_axes] + (w_shape[0],)
    x = x.reshape(numpy.prod(x_shape[:n_batch_axes]),
                  numpy.prod(x_shape[n_batch_axes:]))
    numpy_out = x.dot(w.T).reshape(out_shape)
    if b is not None:
        numpy_out += b

    chainerx.testing.assert_allclose_ex(
        chainerx_out, numpy_out,
        float16_rtol=1e-2, float16_atol=1e-2, strides_check=False)
