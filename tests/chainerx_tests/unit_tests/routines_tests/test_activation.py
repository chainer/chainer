import numpy
import pytest

import chainerx
import chainerx.testing

from chainerx_tests import array_utils


@chainerx.testing.numpy_chainerx_array_equal()
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_relu(xp, device, shape, dtype):
    if dtype == 'bool_':
        return chainerx.testing.ignore()
    x = array_utils.create_dummy_ndarray(xp, shape, dtype)
    if xp is numpy:
        return numpy.maximum(0, x)
    else:
        return chainerx.relu(x)


@chainerx.testing.numpy_chainerx_allclose(atol=1e-6)
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_sigmoid(xp, device, shape, dtype):
    if dtype in ['bool_', 'uint8', 'int8', 'int16', 'int32', 'int64']:
        return chainerx.testing.ignore()
    x = array_utils.create_dummy_ndarray(xp, shape, dtype)
    if xp is numpy:
        return numpy.reciprocal(1 + numpy.exp(-x))
    else:
        return chainerx.sigmoid(x)
