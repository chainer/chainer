import unittest

import mock
import numpy

import chainer
from chainer import testing


class TestFunctionHook(unittest.TestCase):

    def setUp(self):
        self.h = chainer.FunctionHook()

    def test_name(self):
        self.assertEqual(self.h.name, 'FunctionHook')

    def test_forward_preprocess(self):
        self.assertTrue(hasattr(self.h, 'forward_preprocess'))

    def test_forward_postprocess(self):
        self.assertTrue(hasattr(self.h, 'forward_postprocess'))

    def test_backward_preprocess(self):
        self.assertTrue(hasattr(self.h, 'backward_preprocess'))

    def test_backward_postprocess(self):
        self.assertTrue(hasattr(self.h, 'backward_postprocess'))

    def check_hook_methods_called(self, func):
        def check_method_called(name):
            with mock.patch.object(self.h, name) as patched:
                with self.h:
                    func()
                patched.assert_called()

        check_method_called('forward_preprocess')
        check_method_called('forward_postprocess')
        check_method_called('backward_preprocess')
        check_method_called('backward_postprocess')

    def test_all_called_with_backward(self):
        x = chainer.Variable(numpy.random.rand(2, 3).astype(numpy.float32))
        y = chainer.functions.sum(x * x)
        self.check_hook_methods_called(y.backward)

    def test_all_called_with_grad(self):
        x = chainer.Variable(numpy.random.rand(2, 3).astype(numpy.float32))
        y = chainer.functions.sum(x * x)
        self.check_hook_methods_called(lambda: chainer.grad([y], [x]))


testing.run_module(__name__, __file__)
