import unittest

import chainer
from chainer import cuda
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


testing.run_module(__name__, __file__)
