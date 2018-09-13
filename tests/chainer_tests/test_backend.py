import unittest

import numpy
import pytest

from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import testing
from chainer.testing import attr
import chainerx


class TestToNumpy(unittest.TestCase):

    def setUp(self):
        self.array = numpy.arange(1, 5, dtype=numpy.float32)

    def check_to_numpy(self, array):
        array_numpy = backend.to_numpy(array)
        assert self.array.dtype == array_numpy.dtype
        assert numpy.array_equal(self.array, array_numpy)

    @pytest.mark.chainerx
    def test_to_numpy_chainerx(self):
        array = chainerx.asarray(self.array)
        assert isinstance(array, chainerx.ndarray)
        self.check_to_numpy(array)

    def test_to_numpy_cpu(self):
        assert isinstance(self.array, numpy.ndarray)
        self.check_to_numpy(self.array)

    @attr.gpu
    def test_to_numpy_gpu(self):
        array = cuda.to_gpu(self.array)
        assert isinstance(array, cuda.ndarray)
        self.check_to_numpy(array)

    @attr.ideep
    def test_to_numpy_ideep(self):
        array = intel64.ideep.array(self.array)
        assert isinstance(array, intel64.mdarray)
        self.check_to_numpy(array)


class TestToNumpyNone(unittest.TestCase):

    def test_to_numpy_empty(self):
        array_numpy = backend.to_numpy(None)
        assert array_numpy is None


class TestToNumpyIterable(unittest.TestCase):

    def test_to_numpy_iterable(self):
        arrays = numpy.split(
            numpy.arange(8).reshape(4, 2).astype(numpy.float32), 2)
        arrays_numpy = backend.to_numpy(arrays)
        assert type(arrays) == type(arrays_numpy)
        for a, na in zip(arrays, arrays_numpy):
            assert a.dtype == na.dtype
            assert numpy.array_equal(a, na)

    def test_to_numpy_empty_list(self):
        arrays_numpy = backend.to_numpy([])
        assert type(arrays_numpy) is list
        assert 0 == len(arrays_numpy)

    def test_to_numpy_empty_tuple(self):
        arrays_numpy = backend.to_numpy(())
        assert type(arrays_numpy) is tuple
        assert 0 == len(arrays_numpy)


testing.run_module(__name__, __file__)
