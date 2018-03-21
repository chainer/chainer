import os
import unittest

import chainer
from chainer import initializers
from chainer import testing

import numpy


class TestGenerateArray(unittest.TestCase):

    def _generate_array(self, dtype=None):
        initializer = initializers.Zero(dtype)
        return initializers.generate_array(initializer, (), numpy)

    def test_default_init(self):
        default_dtype = os.environ.get('CHAINER_DTYPE', 'float32')
        array = self._generate_array()
        self.assertEqual(default_dtype, array.dtype)

    def test_custom_init(self):
        with chainer.using_config('dtype', 'float16'):
            array = self._generate_array()
        self.assertEqual('float16', array.dtype)

    def test_init_with_initializer_dtype(self):
        with chainer.using_config('dtype', 'float16'):
            array = self._generate_array('float64')
        self.assertEqual('float64', array.dtype)


class TestGetInitializer(unittest.TestCase):

    def test_scalar(self):
        init = initializers._get_initializer(10)
        self.assertIsInstance(init, initializers.Constant)

        x = numpy.empty((2, 3), dtype=numpy.int32)
        init(x)

        expected = numpy.full((2, 3), 10, dtype=numpy.int32)
        numpy.testing.assert_array_equal(x, expected)

    def test_numpy_array(self):
        c = numpy.array([1, 2, 3])
        init = initializers._get_initializer(c)

        self.assertIsInstance(init, initializers.Constant)

        x = numpy.empty((3,), dtype=numpy.int32)
        init(x)

        expected = numpy.array([1, 2, 3], dtype=numpy.int32)
        numpy.testing.assert_array_equal(x, expected)

    def test_callable(self):

        def initializer(arr):
            arr[...] = 100

        init = initializers._get_initializer(initializer)
        self.assertTrue(callable(init))

        x = numpy.empty((2, 3), dtype=numpy.int32)
        init(x)

        expected = numpy.full((2, 3), 100, dtype=numpy.int32)
        numpy.testing.assert_array_equal(x, expected)


testing.run_module(__name__, __file__)
