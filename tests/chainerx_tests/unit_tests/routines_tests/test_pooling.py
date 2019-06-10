import unittest

import chainer
import numpy
import pytest

import chainerx

from chainerx_tests import op_utils


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('x_shape,ksize,stride,pad', [
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
@chainer.testing.parameterize_pytest('cover_all', [True, False])
class TestMaxPool(op_utils.ChainerOpTest):

    dodge_nondifferentiable = True

    def setup(self, float_dtype):
        dtype = float_dtype
        ksize = self.ksize
        device = chainerx.get_default_device()
        if (device.backend.name == 'cuda'
                and len(ksize) != 2
                and len(ksize) != 3):
            raise unittest.SkipTest(
                'cuDNN supports only 2 and 3 spatial dimensions')

        if dtype == 'float16':
            self.check_backward_options.update({'rtol': 5e-2, 'atol': 1e-3})
            self.check_double_backward_options.update({
                'rtol': 5e-2, 'atol': 1e-3})

        self.dtype = dtype

    def generate_inputs(self):
        x_shape = self.x_shape
        dtype = self.dtype
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        return x,

    def forward_chainerx(self, inputs):
        x, = inputs
        y = chainerx.max_pool(
            x, ksize=self.ksize, stride=self.stride, pad=self.pad,
            cover_all=self.cover_all)

        # This function can return -inf (or huge negative numbers in case of
        # CUDA) around boundaries.
        # Convert them to finite numbers in order to properly calculate numeric
        # gradients.
        y = chainerx.maximum(y, -1e4)
        return y,

    def forward_chainer(self, inputs):
        x, = inputs
        y = chainer.functions.max_pooling_nd(
            x, ksize=self.ksize, stride=self.stride, pad=self.pad,
            cover_all=self.cover_all)
        # Convert -inf to finite numbers.
        y = chainer.functions.maximum(y, numpy.full_like(y.array, -1e4))
        return y,


@pytest.mark.parametrize('x_shape,ksize,stride,pad', [
    ((1, 3), (), 1, 0),               # Requires at least one spatial dimension
    ((2, 3, 4, 3), (2, 2, 1), 3, 2),  # Wrong number of ksize.
    ((2, 3, 4, 3), (2, 2), (1,), 0),  # Wrong number of strides.
    ((1, 3, 4, 3), (2, 2), 3, (2,)),  # Wrong number of paddings.
    ((4, 4, 2, 2), 5, 3, 0),          # Output size should be positive.
])
@pytest.mark.parametrize('cover_all', [True, False])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_max_pool_invalid(
        device, x_shape, ksize, stride, pad, cover_all, float_dtype):
    x = numpy.random.uniform(-1, 1, x_shape).astype(float_dtype)
    x = chainerx.array(x)
    with pytest.raises(chainerx.DimensionError):
        chainerx.max_pool(
            x, ksize=ksize, stride=stride, pad=pad, cover_all=cover_all)


def _get_pad_mode_kwargs(pad_mode, is_chainerx):
    # ChainerX
    if is_chainerx:
        if pad_mode is None:
            return {}
        return {'pad_mode': pad_mode}
    # Chainer
    # chainerx `pad_mode` defaults to 'ignore', whereas chainer's default is
    # pad_value=0.
    if pad_mode == 'zero':
        return {'pad_value': 0}
    if pad_mode in ('ignore', None):
        return {'pad_value': None}
    assert False, pad_mode


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize_pytest('x_shape,ksize,stride,pad', [
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
@chainer.testing.parameterize_pytest('pad_mode', ['zero', 'ignore', None])
# ignore warning occuring when pad_value is None in chainer
@pytest.mark.filterwarnings('ignore:invalid value encountered in true_divide')
class TestAveragePool(op_utils.ChainerOpTest):

    def setup(self, float_dtype):
        dtype = float_dtype
        ksize = self.ksize
        device = chainerx.get_default_device()
        if (device.backend.name == 'cuda'
                and len(ksize) != 2
                and len(ksize) != 3):
            raise unittest.SkipTest(
                'cuDNN supports only 2 and 3 spatial dimensions.')

        # TODO(niboshi): average_pool can return nan if pad_mode is 'ignore',
        # and numeric gradients cannot be calculated.
        # If chainerx.where is implemented, we can replace nans and remove
        # this skip.
        if self.pad_mode in ('ignore', None):
            self.skip_backward_test = True
            self.skip_double_backward_test = True

        self.check_double_backward_options.update({'rtol': 5e-3, 'atol': 5e-3})
        if dtype == 'float16':
            self.check_forward_options.update({'rtol': 5e-3, 'atol': 5e-4})
            self.check_backward_options.update({'rtol': 5e-2, 'atol': 5e-3})
        else:
            self.check_backward_options.update({'rtol': 5e-3, 'atol': 5e-3, })

        self.dtype = dtype

    def generate_inputs(self):
        x_shape = self.x_shape
        dtype = self.dtype
        x = numpy.random.uniform(-1, 1, x_shape).astype(dtype)
        return x,

    def forward_chainerx(self, inputs):
        x, = inputs
        pad_mode_kwargs = _get_pad_mode_kwargs(self.pad_mode, True)
        y = chainerx.average_pool(
            x, ksize=self.ksize, stride=self.stride, pad=self.pad,
            **pad_mode_kwargs)
        return y,

    def forward_chainer(self, inputs):
        x, = inputs
        pad_value_kwargs = _get_pad_mode_kwargs(self.pad_mode, False)
        y = chainer.functions.average_pooling_nd(
            x, ksize=self.ksize, stride=self.stride, pad=self.pad,
            **pad_value_kwargs)
        return y,


@pytest.mark.parametrize('x_shape,ksize,stride,pad', [
    ((1, 3), (), 1, 0),               # Requires at least one spatial dimension
    ((2, 3, 4, 3), (2, 2, 1), 3, 2),  # Wrong number of ksize.
    ((2, 3, 4, 3), (2, 2), (1,), 0),  # Wrong number of strides.
    ((1, 3, 4, 3), (2, 2), 3, (2,)),  # Wrong number of paddings.
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('pad_mode', ['zero', 'ignore', None])
def test_average_pool_invalid(
        device, x_shape, ksize, stride, pad, pad_mode, float_dtype):
    x = numpy.random.uniform(-1, 1, x_shape).astype(float_dtype)
    x = chainerx.array(x)
    pad_mode_kwargs = _get_pad_mode_kwargs(pad_mode, True)
    with pytest.raises(chainerx.DimensionError):
        chainerx.average_pool(
            x, ksize=ksize, stride=stride, pad=pad, **pad_mode_kwargs)
