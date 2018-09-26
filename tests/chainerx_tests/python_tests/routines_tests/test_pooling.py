import chainer
import numpy
import pytest

import chainerx

from chainerx_tests import array_utils


def _create_max_pool_args(xp, device, x_shape, ksize, stride, pad, cover_all, float_dtype):
    x = array_utils.create_dummy_ndarray(xp, x_shape, float_dtype)
    if device.backend.name == 'cuda':  # cover_all is not supported by CUDA.
        cover_all = False
    ret_args = dict(x=x, ksize=ksize)
    if stride is not None:
        ret_args['stride'] = stride
    if pad is not None:
        ret_args['pad'] = pad
    if cover_all is not None:
        ret_args['cover_all'] = cover_all
    return ret_args


@pytest.mark.parametrize('x_shape,ksize,stride,pad', [
    ((2, 3, 4), (1,), 1, 0),
    ((1, 3, 4), (2, ), 3, 2),
    ((1, 3, 4), (2,), 3, 2),
    ((2, 3, 4, 4), (3, 3), 1, 0),
    ((2, 3, 4, 4), (3, 3), None, 0),
    ((1, 3, 4, 4), (3, 3), (1, 2), 1),
    ((1, 3, 4, 4), (3, 3), 2, (2, 0)),
    ((1, 3, 2, 6, 3), (1, 3, 2), 2, (2, 0, 1)),
    ((1, 3, 2, 6, 3), (1, 3, 2), (1, 2, 3), (2, 0, 1)),
    ((2, 3, 2, 6, 3), (1, 3, 2), (1, 2, 3), (2, 0, 1)),
    ((1, 3, 2, 6, 3, 2), (1, 3, 2, 2), 2, 2),
])
@pytest.mark.parametrize('cover_all', [True, False])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_max_pool(device, x_shape, ksize, stride, pad, cover_all, float_dtype):
    if device.backend.name == 'cuda' and len(ksize) != 2 and len(ksize) != 3:
        # cuDNN supports only 2 and 3 spatial dimensions.
        return chainerx.testing.ignore()

    def create_args(xp):
        return _create_max_pool_args(xp, device, x_shape, ksize, stride, pad, cover_all, float_dtype)

    def chainerx_max_pool():
        y = chainerx.max_pool(**create_args(chainerx))
        # In the case of CUDA, we get huge negative numbers instead of -inf around boundaries.
        # Align them to chainer (native) results.
        if device.backend.name == 'cuda':
            y = chainerx.to_numpy(y)
            y[y < -3.e+34] = -float('inf')
            y = chainerx.array(y)
        return y

    chainerx.testing.assert_allclose(chainerx_max_pool(), chainer.functions.max_pooling_nd(**create_args(numpy)).data)


@pytest.mark.parametrize('x_shape,ksize,stride,pad', [
    ((1, 3), (), 1, 0),               # Requires at least one spatial dimension
    ((2, 3, 4, 3), (2, 2, 1), 3, 2),  # Wrong number of ksize.
    ((2, 3, 4, 3), (2, 2), (1,), 0),  # Wrong number of strides.
    ((1, 3, 4, 3), (2, 2), 3, (2,)),  # Wrong number of paddings.
])
@pytest.mark.parametrize('cover_all', [True, False])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_max_pool_invalid(device, x_shape, ksize, stride, pad, cover_all, float_dtype):
    with pytest.raises(chainerx.DimensionError):
        chainerx.max_pool(**_create_max_pool_args(chainerx, device, x_shape, ksize, stride, pad, cover_all, float_dtype))


def _create_average_pool_args(xp, device, x_shape, ksize, stride, pad, pad_mode, float_dtype):
    x = array_utils.create_dummy_ndarray(xp, x_shape, float_dtype)
    ret_args = dict(x=x, ksize=ksize)
    if stride is not None:
        ret_args['stride'] = stride
    if pad is not None:
        ret_args['pad'] = pad

    if pad_mode is None:
        # chainerx defaults to 'ignore', which is equivalent with pad_value=None in chainer
        if xp is not chainerx:
            ret_args['pad_value'] = None
    else:
        if xp is chainerx:
            ret_args['pad_mode'] = pad_mode
        else:
            if pad_mode == 'zero':
                ret_args['pad_value'] = 0
            elif pad_mode == 'ignore':
                ret_args['pad_value'] = None
            else:
                assert False  # should never reach

    return ret_args


@pytest.mark.filterwarnings('ignore:invalid value encountered in true_divide')  # ignore warning occuring when pad_value is None in chainer
@pytest.mark.parametrize('x_shape,ksize,stride,pad', [
    ((2, 3, 4), (1,), 1, 0),
    ((1, 3, 4), (2, ), 3, 2),
    ((1, 3, 4), (2,), 3, 2),
    ((2, 3, 4, 4), (3, 3), 1, 0),
    ((2, 3, 4, 4), (3, 3), None, 0),
    ((1, 3, 4, 4), (3, 3), (1, 2), 1),
    ((1, 3, 4, 4), (3, 3), 2, (2, 0)),
    ((1, 3, 2, 6, 3), (1, 3, 2), 2, (2, 0, 1)),
    ((1, 3, 2, 6, 3), (1, 3, 2), (1, 2, 3), (2, 0, 1)),
    ((2, 3, 2, 6, 3), (1, 3, 2), (1, 2, 3), (2, 0, 1)),
    ((1, 3, 2, 6, 3, 2), (1, 3, 1, 1), 1, 1),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('pad_mode', ['zero', 'ignore', None])
def test_average_pool(device, x_shape, ksize, stride, pad, pad_mode, float_dtype):
    if device.backend.name == 'cuda' and len(ksize) != 2 and len(ksize) != 3:
        # cuDNN supports only 2 and 3 spatial dimensions.
        return chainerx.testing.ignore()

    def create_args(xp):
        return _create_average_pool_args(xp, device, x_shape, ksize, stride, pad, pad_mode, float_dtype)

    chainerx.testing.assert_allclose(chainerx.average_pool(**create_args(chainerx)),
                                     chainer.functions.average_pooling_nd(**create_args(numpy)).data)


@pytest.mark.parametrize('x_shape,ksize,stride,pad', [
    ((1, 3), (), 1, 0),               # Requires at least one spatial dimension
    ((2, 3, 4, 3), (2, 2, 1), 3, 2),  # Wrong number of ksize.
    ((2, 3, 4, 3), (2, 2), (1,), 0),  # Wrong number of strides.
    ((1, 3, 4, 3), (2, 2), 3, (2,)),  # Wrong number of paddings.
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('pad_mode', ['zero', 'ignore', None])
def test_average_pool_invalid(device, x_shape, ksize, stride, pad, pad_mode, float_dtype):
    with pytest.raises(chainerx.DimensionError):
        chainerx.average_pool(**_create_average_pool_args(chainerx, device, x_shape, ksize, stride, pad, pad_mode, float_dtype))
