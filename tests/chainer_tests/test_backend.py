import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import testing
from chainer.testing import attr
import chainerx


class _TestCopyToBase(object):

    src_data = numpy.arange(1, 5, dtype=numpy.float32)
    dst_data = numpy.zeros_like(src_data)

    def _get_dst(self):
        raise NotImplementedError

    def test_from_cpu(self):
        src = self.src_data
        dst = self._get_dst()
        backend.copyto(dst, src)
        numpy.testing.assert_array_equal(cuda.to_cpu(dst), self.src_data)

    @attr.gpu
    def test_from_gpu(self):
        src = cuda.cupy.array(self.src_data)
        dst = self._get_dst()
        backend.copyto(dst, src)
        numpy.testing.assert_array_equal(cuda.to_cpu(dst), self.src_data)

    @attr.ideep
    def test_from_ideep(self):
        src = intel64.ideep.array(self.src_data)
        dst = self._get_dst()
        assert isinstance(src, intel64.mdarray)
        backend.copyto(dst, src)
        numpy.testing.assert_array_equal(cuda.to_cpu(dst), self.src_data)


class TestCopyToCPU(_TestCopyToBase, unittest.TestCase):
    def _get_dst(self):
        return self.dst_data


@attr.gpu
class TestCopyToGPU(_TestCopyToBase, unittest.TestCase):
    def _get_dst(self):
        return cuda.cupy.array(self.dst_data)

    @attr.multi_gpu(2)
    def test_gpu_to_another_gpu(self):
        src = cuda.cupy.array(self.src_data)
        with cuda.get_device_from_id(1):
            dst = self._get_dst()
        backend.copyto(dst, src)
        cuda.cupy.testing.assert_array_equal(dst, src)


@attr.ideep
class TestCopyToIDeep(_TestCopyToBase, unittest.TestCase):
    def _get_dst(self):
        dst = intel64.ideep.array(self.src_data)
        assert isinstance(dst, intel64.mdarray)
        return dst


class TestCopyToError(unittest.TestCase):
    def test_fail_on_invalid_src(self):
        src = None
        dst = numpy.zeros(1)
        with self.assertRaises(TypeError):
            backend.copyto(dst, src)

    def test_fail_on_invalid_dst(self):
        src = numpy.zeros(1)
        dst = None
        with self.assertRaises(TypeError):
            backend.copyto(dst, src)


class TestGetArrayModule(unittest.TestCase):

    def test_get_array_module_for_numpy_array(self):
        xp = backend.get_array_module(numpy.array([]))
        self.assertIs(xp, numpy)
        assert xp is not cuda.cupy
        assert xp is not chainerx

    def test_get_array_module_for_numpy_variable(self):
        xp = backend.get_array_module(chainer.Variable(numpy.array([])))
        assert xp is numpy
        assert xp is not cuda.cupy
        assert xp is not chainerx

    @attr.gpu
    def test_get_array_module_for_cupy_array(self):
        xp = backend.get_array_module(cuda.cupy.array([]))
        assert xp is cuda.cupy
        assert xp is not numpy
        assert xp is not chainerx

    @attr.gpu
    def test_get_array_module_for_cupy_variable(self):
        xp = backend.get_array_module(chainer.Variable(cuda.cupy.array([])))
        assert xp is cuda.cupy
        assert xp is not numpy
        assert xp is not chainerx


    @attr.chainerx
    def test_get_array_module_for_chainerx_array(self):
        xp = backend.get_array_module(chainerx.array([]))
        assert xp is chainerx
        assert xp is not numpy
        assert xp is not cuda.cupy

    @attr.chainerx
    def test_get_array_module_for_chainerx_variable(self):
        xp = backend.get_array_module(chainer.Variable(chainerx.array([])))
        assert xp is chainerx
        assert xp is not numpy
        assert xp is not cuda.cupy


testing.run_module(__name__, __file__)
