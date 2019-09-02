import chainer
from chainer import functions as F
import numpy
import pytest

import chainerx

from chainerx_tests import array_utils
from chainerx_tests import op_utils


_in_out_eval_dtypes = [
    (('float16', 'int16')),
    (('float32', 'int32')),
    (('float64', 'int64')),
    (('float32', 'int16')),
    (('float64', 'int16')),
    (('float64', 'int32')),
]


_accuracy_params = [
    ((10, 1), (10,)),
    ((5, 1), (5,)),
    ((10, 3), (10,)),
    ((10, 3, 1), (10,)),
    ((10, 3, 1, 1), (10,)),
    ((10, 3, 5), (10, 5)),
    ((10, 3, 5, 4), (10, 5, 4)),
    ((10, 3, 5, 4, 1), (10, 5, 4)),
    ((10, 3, 5, 4, 1, 1), (10, 5, 4)),
]


_invalid_accuracy_dtypes = [
    (('int16', 'float16')),
    (('int32', 'int32')),
    (('float32', 'float32')),
    (('float64', 'float64')),
    (('int64', 'float64')),
]


_invalid_accuracy_shapes = [
    ((10, 1), (5,)),
    ((5, 3), (10, 3)),
]


class EvalBase(op_utils.ChainerOpTest):

    def generate_inputs(self):
        y_dtype, t_dtype = self.in_dtypes
        y = numpy.random.uniform(-1, 1, self.y_shape).astype(y_dtype)
        targ = numpy.random.randint(
            3, size=self.t_shape).astype(t_dtype)
        return y, targ

    def forward_chainerx(self, inputs):
        return self.forward_xp(inputs, chainerx)

    def forward_chainer(self, inputs):
        return self.forward_xp(inputs, F)

    def forward_xp(self, inputs, xp):
        raise NotImplementedError(
            'Op test implementation must override `forward_xp`.')


@op_utils.op_test(['native:0', 'cuda:0'])
@chainer.testing.parameterize(*(
    chainer.testing.product([
        chainer.testing.from_pytest_parameterize(
            'y_shape,t_shape', _accuracy_params),
        chainer.testing.from_pytest_parameterize(
            'in_dtypes', _in_out_eval_dtypes),
        chainer.testing.from_pytest_parameterize(
            'ignore_label', [None, 0])
    ])
))
class TestAccuracy(EvalBase):

    skip_backward_test = True
    skip_double_backward_test = True

    def setup(self):
        super().setup()
        dtype1, dtype2 = self.in_dtypes
        if dtype1 == 'float16' or dtype2 == 'float16':
            self.check_forward_options.update({'rtol': 1e-2, 'atol': 1e-2})

    def generate_inputs(self):
        y, t = super().generate_inputs()
        # TODO(aksub99): Improve tests for the case
        # where all labels are ignored.
        if y.shape == (10, 1) or y.shape == (5, 1):
            self.ignore_label = 0
            t.fill(self.ignore_label)
        return y, t

    def forward_xp(self, inputs, xp):
        y, t = inputs
        out = xp.accuracy(y, t, self.ignore_label)
        return out,


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('y_shape,t_shape', _accuracy_params)
@pytest.mark.parametrize('in_dtypes', _invalid_accuracy_dtypes)
@pytest.mark.parametrize('ignore_label', [None, 0])
def test_accuracy_invalid_dtype(device, y_shape,
                                t_shape, ignore_label, in_dtypes):
    dtype1, dtype2 = in_dtypes
    y = array_utils.create_dummy_ndarray(chainerx, y_shape, dtype1)
    t = array_utils.create_dummy_ndarray(chainerx, t_shape, dtype2)
    with pytest.raises(chainerx.DtypeError):
        chainerx.accuracy(y, t, ignore_label=ignore_label)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('y_shape,t_shape', _invalid_accuracy_shapes)
@pytest.mark.parametrize('in_dtypes', _in_out_eval_dtypes)
@pytest.mark.parametrize('ignore_label', [None, 0])
def test_accuracy_invalid_shape(device, y_shape,
                                t_shape, ignore_label, in_dtypes):
    dtype1, dtype2 = in_dtypes
    y = array_utils.create_dummy_ndarray(chainerx, y_shape, dtype1)
    t = array_utils.create_dummy_ndarray(chainerx, t_shape, dtype2)
    with pytest.raises(chainerx.DimensionError):
        chainerx.accuracy(y, t, ignore_label=ignore_label)
