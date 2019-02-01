import unittest

from chainer import function_node
from chainer import testing


def dummy():
    pass


class TestNoNumpyFunction(unittest.TestCase):

    def test_no_numpy_function(self):
        with self.assertRaises(ValueError):
            testing.unary_math_function_unittest(dummy)  # no numpy.dummy


class DummyLinear(function_node.FunctionNode):

    @property
    def label(self):
        return 'dummy_linear'

    def forward(self, x):
        return x[0],

    def backward(self, indexes, gy):
        return gy[0],


def dummy_linear(x):
    return DummyLinear().apply((x,))[0]


@testing.unary_math_function_unittest(dummy_linear,
                                      func_expected=lambda x, dtype: x)
class TestIsLinear(unittest.TestCase):
    pass


testing.run_module(__name__, __file__)
