import unittest

import numpy

import chainerx
import chainerx.testing

from chainerx_tests import array_utils
from chainerx_tests import op_utils


@op_utils.op_test(['native:0', 'cuda:0'])
class TestRelu(op_utils.OpTest):

    dodge_nondifferentiable = True

    def setup(self, shape, dtype):
        if dtype == 'bool_':
            raise unittest.SkipTest('bool is not supported')
        # Skip backward/double-backward tests for int dtypes
        if numpy.dtype(dtype).kind != 'f':
            self.skip_backward_test = True
            self.skip_double_backward_test = True

        self.shape = shape
        self.dtype = dtype

    def generate_inputs(self):
        shape = self.shape
        dtype = self.dtype
        x = numpy.random.uniform(-1, 1, shape).astype(dtype)
        return x,

    def forward_chainerx(self, inputs):
        x, = inputs
        y = chainerx.relu(x)
        return y,

    def forward_expected(self, inputs):
        x, = inputs
        y = numpy.asarray(numpy.maximum(0, x)).astype(x.dtype)
        return y,


@op_utils.op_test(['native:0', 'cuda:0'])
class TestSigmoid(op_utils.OpTest):

    # TODO(imanishi): Dtype promotion is not supported yet.
    def setup(self, shape, float_dtype):
        self.shape = shape
        self.dtype = float_dtype

        if float_dtype == 'float16':
            self.check_forward_options.update({'atol': 1e-4, 'rtol': 1e-3})
            self.check_backward_options.update({'atol': 1e-2, 'rtol': 5e-2})
            self.check_double_backward_options.update(
                {'atol': 1e-2, 'rtol': 5e-2})

    def generate_inputs(self):
        shape = self.shape
        dtype = self.dtype
        x = array_utils.create_dummy_ndarray(numpy, shape, dtype)
        return x,

    def forward_chainerx(self, inputs):
        x, = inputs
        y = chainerx.sigmoid(x)
        return y,

    def forward_expected(self, inputs):
        x, = inputs
        y = numpy.asarray(numpy.reciprocal(1 + numpy.exp(-x))).astype(x.dtype)
        return y,
