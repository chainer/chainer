import unittest

import numpy
import pytest

from chainer import testing


# TODO(niboshi): Add more assert_allclose tests


class TestAssertAllClose(unittest.TestCase):

    def test_no_zero_division(self):
        # No zero-division should occur when the relative error is inf (y=0).
        # That would cause FloatingPointError with
        # numpy.errstate(divide='raise').
        with numpy.errstate(divide='raise'):
            with pytest.raises(AssertionError):
                x = numpy.array([1], numpy.float32)
                y = numpy.array([0], numpy.float32)
                testing.assert_allclose(x, y)


testing.run_module(__name__, __file__)
