import unittest

import cupy
from cupy import cuda
from cupy import testing
import numpy
from numpy import testing as np_testing


@testing.gpu
class TestArrayGet(unittest.TestCase):

    _multiprocess_can_split_ = True

    def setUp(self):
        self.stream = cuda.Stream()

    def check_get(self, f, stream):
        a_gpu = f(cupy)
        a_cpu = f(numpy)
        np_testing.assert_array_equal(a_gpu.get(stream), a_cpu)

    @testing.for_all_dtypes()
    def test_contiguous_array(self, dtype):
        def contiguous_array(xp):
            return testing.shaped_arange((3,), xp=xp, dtype=dtype)
        self.check_get(contiguous_array, None)

    @testing.for_all_dtypes()
    def test_non_contiguous_array(self, dtype):
        def non_contiguous_array(xp):
            return testing.shaped_arange((3,), xp=xp, dtype=dtype)[0::2]
        self.check_get(non_contiguous_array, None)

    @testing.for_all_dtypes()
    def test_contiguous_array_stream(self, dtype):
        def contiguous_array(xp):
            return testing.shaped_arange((3,), xp=xp, dtype=dtype)
        self.check_get(contiguous_array, self.stream.ptr)

    @testing.for_all_dtypes()
    def test_non_contiguous_array_stream(self, dtype):
        def non_contiguous_array(xp):
            return testing.shaped_arange((3,), xp=xp, dtype=dtype)[0::2]
        self.check_get(non_contiguous_array, self.stream.ptr)
