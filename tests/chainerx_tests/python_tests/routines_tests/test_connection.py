import chainer
import numpy
import pytest

import chainerx

from chainerx_tests import array_utils


def _create_conv_args(xp, device, x_shape, w_shape, b_shape, stride, pad, cover_all, float_dtype):
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
def test_conv(device, x_shape, w_shape, b_shape, stride, pad, cover_all, float_dtype):
    if device.backend.name == 'cuda' and len(x_shape) <= 3:
        # cuDNN does not support 1 dimensional convolution and throws DimensionError.
        # TODO(hvy): Support 1 dimensional convolution with CUDA.
        return chainerx.testing.ignore()

    def create_args(xp):
        return _create_conv_args(xp, device, x_shape, w_shape, b_shape, stride, pad, cover_all, float_dtype)
    chainerx.testing.assert_allclose(chainerx.conv(*create_args(chainerx)), chainer.functions.convolution_nd(*create_args(numpy)).data)


@pytest.mark.parametrize('x_shape,w_shape,b_shape,stride,pad', [
    ((1, 3, 4, 3), (5, 4, 2, 2), (5,), 3, 2),  # Mismatched x and w input channels.
    ((2, 3, 4, 3), (5, 3, 2, 2, 1), (5,), 3, 2),  # Mismatched x and w dimensions.
    ((1, 3, 4, 3), (5, 3, 2, 2), (6,), 1, 0),  # Mismatched w and b.
    ((2, 3, 4, 3), (5, 3, 2, 2), None, (1,), 0),  # Wrong number of strides.
    ((1, 3, 4, 3), (5, 3, 2, 2), None, 3, (2,)),  # Wrong number of paddings.
])
@pytest.mark.parametrize('cover_all', [True, False])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_conv_invalid(device, x_shape, w_shape, b_shape, stride, pad, cover_all, float_dtype):
    with pytest.raises(chainerx.DimensionError):
        chainerx.conv(*_create_conv_args(chainerx, device, x_shape, w_shape, b_shape, stride, pad, cover_all, float_dtype))


def _get_conv_transpose_outsize(x_shape, w_shape, stride, pad, cover_all):
    in_dims = x_shape[2:]
    kernel_size = w_shape[2:]
    ndim = len(in_dims)
    stride_tup = (stride,) * ndim if isinstance(stride, int) else stride
    pad_tup = (pad,) * ndim if isinstance(pad, int) else pad
    return tuple(chainer.utils.conv.get_deconv_outsize(d, k, s, p, cover_all) for (d, k, s, p)
                 in zip(in_dims, kernel_size, stride_tup, pad_tup))


def _create_conv_transpose_args(xp, device, x_shape, w_shape, b_shape, stride, pad, outsize, float_dtype):
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
@pytest.mark.parametrize('cover_all', [None, True, False])  # If None, outsize argument will be None.
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_conv_transpose(device, x_shape, w_shape, b_shape, stride, pad, cover_all, float_dtype):
    if device.backend.name == 'cuda' and len(x_shape) <= 3:
        # cuDNN does not support 1 dimensional convolution and throws DimensionError.
        # TODO(sonots): Support 1 dimensional convolution with CUDA.
        return chainerx.testing.ignore()
    if device.backend.name == 'cuda' and cover_all is True:
        # outsize (for cover_all=True) is not supported by CUDA.
        return chainerx.testing.ignore()

    def create_args(xp):
        if cover_all is None:
            outsize = None
        else:
            outsize = _get_conv_transpose_outsize(x_shape, w_shape, stride, pad, cover_all)
        return _create_conv_transpose_args(xp, device, x_shape, w_shape, b_shape, stride, pad, outsize, float_dtype)

    chainerx.testing.assert_allclose(chainerx.conv_transpose(*create_args(chainerx)),
                                     chainer.functions.deconvolution_nd(*create_args(numpy)).data)


@pytest.mark.parametrize('x_shape,w_shape,b_shape,stride,pad,outsize', [
    ((1, 3, 4, 3), (5, 4, 2, 2), (5,), 3, 2, None),  # Mismatched x and w input channels.
    ((2, 3, 4, 3), (3, 5, 2, 2, 1), (5,), 3, 2, None),  # Mismatched x and w dimensions.
    ((1, 3, 4, 3), (3, 5, 2, 2), (6,), 1, 0, None),  # Mismatched w and b.
    ((2, 3, 4, 3), (3, 5, 2, 2), None, (1,), 0, None),  # Wrong number of strides.
    ((1, 3, 4, 3), (3, 5, 2, 2), None, 3, (2,), None),  # Wrong number of paddings.
    ((1, 3, 2, 6, 3), (3, 2, 1, 3, 2), (2,), 2, (2, 0, 1), (-1, 13, 4)),  # All output sizes must be non-negative
    ((1, 3, 2, 6, 3), (3, 2, 1, 3, 2), (2,), 2, (2, 0, 1), None),  # All output sizes must be non-negative
    ((2, 3, 4), (3, 5, 1), (5,), 1, 0, (5,)),  # Output dims are inconsistent
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_conv_transpose_invalid(device, x_shape, w_shape, b_shape, stride, pad, outsize, float_dtype):
    with pytest.raises(chainerx.DimensionError):
        chainerx.conv_transpose(*_create_conv_transpose_args(
            chainerx, device, x_shape, w_shape, b_shape, stride, pad, outsize, float_dtype))
