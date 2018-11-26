import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import array_utils


# A special parameter object used to represent an unspecified argument.
class Unspecified:
    pass


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize('a_shape,b_shape', [
    ((), ()),
    ((), (2, 3)),
    ((2, 0), (0, 3)),
    ((0, 0), (0, 0)),
    ((2, 3), (3, 4)),
    # TODO(niboshi): Add test cases for more than 2 ndim
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_dot(is_module, xp, device, a_shape, b_shape, dtype):
    # TODO(beam2d): Remove the skip after supporting non-float dot on CUDA
    if device.name == 'cuda:0' and numpy.dtype(dtype).kind != 'f':
        return chainerx.testing.ignore()
    a = array_utils.create_dummy_ndarray(xp, a_shape, dtype)
    b = array_utils.create_dummy_ndarray(xp, b_shape, dtype)
    if is_module:
        return xp.dot(a, b)
    else:
        return a.dot(b)


@chainerx.testing.numpy_chainerx_array_equal(
    accept_error=(chainerx.DimensionError, ValueError))
@pytest.mark.parametrize('a_shape,b_shape', [
    ((3, 2), (1, 3)),
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

    chainerx.testing.assert_array_equal(chainerx_out, numpy_out)
