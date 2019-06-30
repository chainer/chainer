import chainer
from chainer import functions as F
import numpy

import chainerx

from chainerx_tests import dtype_utils
from chainerx_tests import op_utils


_in_out_eval_dtypes = dtype_utils._permutate_dtype_mapping([
    (('float16', 'float16'), 'float16'),
    (('float32', 'float32'), 'float32'),
    (('float64', 'float64'), 'float64'),
    (('float32', 'float16'), 'float32'),
    (('float64', 'float16'), 'float64'),
    (('float64', 'float32'), 'float64'),
])


class EvalBase(op_utils.ChainerOpTest):

    def generate_inputs(self):
        y = numpy.random.normal(loc=0, scale=1.0, size=self.shape)
        targ = numpy.random.normal(loc=0, scale=1.0, size=self.shape) + \
            numpy.random.normal(loc=0, scale=0.5, size=self.shape)
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
            'shape', [
                (2, 2),
                (3, 3, 3),
                (5, 5, 5),
                (4, 1, 2, 4)
            ]),
        chainer.testing.from_pytest_parameterize(
            'in_dtypes,out_dtype', _in_out_eval_dtypes),
        chainer.testing.from_pytest_parameterize(
            'ignore_label', [None, 0])
    ])
))
class TestAccuracy(EvalBase):

    def forward_xp(self, inputs, xp):
        x, t = inputs
        t = t.astype(numpy.int64)
        if xp is chainerx:
            out = xp.accuracy(x, t, self.ignore_label)
        else:
            out = xp.accuracy(x, t, self.ignore_label)
        return out,