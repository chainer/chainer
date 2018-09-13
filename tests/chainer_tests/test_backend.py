import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.testing import attr


class TestGetArrayModule(unittest.TestCase):

    def test_get_array_module_for_numpy_array(self):
        xp = backend.get_array_module(numpy.array([]))
        self.assertIs(xp, numpy)
        assert xp is not cuda.cupy

    def test_get_array_module_for_numpy_variable(self):
        xp = backend.get_array_module(chainer.Variable(numpy.array([])))
        assert xp is numpy
        assert xp is not cuda.cupy

    @attr.gpu
    def test_get_array_module_for_cupy_array(self):
        xp = backend.get_array_module(cuda.cupy.array([]))
        assert xp is cuda.cupy
        assert xp is not numpy

    @attr.gpu
    def test_get_array_module_for_cupy_variable(self):
        xp = backend.get_array_module(chainer.Variable(cuda.cupy.array([])))
        assert xp is cuda.cupy
        assert xp is not numpy


testing.run_module(__name__, __file__)
