import chainer
import numpy
import pytest

import xchainer

from tests import array_utils


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
        return xchainer.testing.ignore()

    def create_args(xp):
        return _create_max_pool_args(xp, device, x_shape, ksize, stride, pad, cover_all, float_dtype)

    def xchainer_max_pool():
        y = xchainer.max_pool(**create_args(xchainer))
        # In the case of CUDA, we get huge negative numbers instead of -inf around boundaries.
        # Align them to chainer (native) results.
        if device.backend.name == 'cuda':
            y = xchainer.tonumpy(y)
            y[y < -3.e+34] = -float('inf')
            y = xchainer.array(y)
        return y

    xchainer.testing.assert_allclose(xchainer_max_pool(), chainer.functions.max_pooling_nd(**create_args(numpy)).data)


@pytest.mark.parametrize('x_shape,ksize,stride,pad', [
    ((1, 3), (), 1, 0),               # Requires at least one spatial dimension
    ((2, 3, 4, 3), (2, 2, 1), 3, 2),  # Wrong number of ksize.
    ((2, 3, 4, 3), (2, 2), (1,), 0),  # Wrong number of strides.
    ((1, 3, 4, 3), (2, 2), 3, (2,)),  # Wrong number of paddings.
])
@pytest.mark.parametrize('cover_all', [True, False])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_max_pool_invalid(device, x_shape, ksize, stride, pad, cover_all, float_dtype):
    with pytest.raises(xchainer.DimensionError):
        xchainer.max_pool(**_create_max_pool_args(xchainer, device, x_shape, ksize, stride, pad, cover_all, float_dtype))
