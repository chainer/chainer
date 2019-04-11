import unittest

import numpy

import chainer
from chainer.backends import cuda
from chainer import testing
from chainer.testing import attr


class TestUseCuDNN(unittest.TestCase):

    @attr.cudnn
    def test_valid_case_combination(self):
        with chainer.using_config('use_cudnn', 'always'):
            self.assertTrue(chainer.should_use_cudnn('==always'))
            self.assertTrue(chainer.should_use_cudnn('>=auto'))

        with chainer.using_config('use_cudnn', 'auto'):
            self.assertFalse(chainer.should_use_cudnn('==always'))
            self.assertTrue(chainer.should_use_cudnn('>=auto'))

        with chainer.using_config('use_cudnn', 'never'):
            self.assertFalse(chainer.should_use_cudnn('==always'))
            self.assertFalse(chainer.should_use_cudnn('>=auto'))

    @unittest.skip(not cuda.cudnn_enabled)
    def test_no_cudnn_available(self):
        with chainer.using_config('use_cudnn', 'always'):
            self.assertFalse(chainer.should_use_cudnn('==always'))
            self.assertFalse(chainer.should_use_cudnn('>=auto'))

    @attr.cudnn
    def test_invalid_level(self):
        self.assertRaises(ValueError, chainer.should_use_cudnn, '==auto')

    @attr.cudnn
    def test_invalid_config(self):
        with chainer.using_config('use_cudnn', True):
            self.assertRaises(ValueError, chainer.should_use_cudnn, '>=auto')

        with chainer.using_config('use_cudnn', False):
            self.assertRaises(ValueError, chainer.should_use_cudnn, '>=auto')

        with chainer.using_config('use_cudnn', 'on'):
            self.assertRaises(ValueError, chainer.should_use_cudnn, '>=auto')

    @attr.cudnn
    def test_higher_version_required(self):
        with chainer.using_config('use_cudnn', 'always'):
            self.assertFalse(chainer.should_use_cudnn(
                '>=auto', cuda.cuda.cudnn.getVersion() + 1))


class TestDtype(unittest.TestCase):

    def test_numpy_dtypes(self):
        for dtype in (numpy.float16, numpy.float32, numpy.float64):
            with chainer.using_config('dtype', dtype):
                self.assertEqual(chainer.get_dtype(), numpy.dtype(dtype))

    def test_specified_dtype(self):
        with chainer.using_config('dtype', numpy.float64):
            dtype = numpy.float16
            self.assertEqual(chainer.get_dtype(dtype), numpy.dtype(dtype))

    def test_mixed16_dtype(self):
        with chainer.using_config('dtype', chainer.mixed16):
            self.assertEqual(chainer.get_dtype(),
                             numpy.dtype(numpy.float16))
            self.assertEqual(chainer.get_dtype(map_mixed16=numpy.float32),
                             numpy.dtype(numpy.float32))

    def test_specified_mixed16_dtype(self):
        with chainer.using_config('dtype', numpy.float64):
            self.assertEqual(chainer.get_dtype(chainer.mixed16),
                             numpy.dtype(numpy.float16))
            self.assertEqual(
                chainer.get_dtype(chainer.mixed16, map_mixed16=numpy.float32),
                numpy.dtype(numpy.float32))


testing.run_module(__name__, __file__)
