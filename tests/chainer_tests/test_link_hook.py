import unittest

import mock
import numpy

import chainer
from chainer import testing


class TestLinkHook(unittest.TestCase):

    def setUp(self):
        self.h = chainer.LinkHook()

        class Model(chainer.Chain):
            def __init__(self):
                super(Model, self).__init__()
                with self.init_scope():
                    self.l1 = chainer.links.Linear(3, 2)

            def forward(self, x):
                return self.l1(x)

        self.model = Model()

    def test_name(self):
        self.assertEqual(self.h.name, 'LinkHook')

    def test_forward_preprocess(self):
        self.assertTrue(hasattr(self.h, 'forward_preprocess'))

    def test_forward_postprocess(self):
        self.assertTrue(hasattr(self.h, 'forward_postprocess'))

    def check_hook_methods_called(self, x):
        def check_method_called(name):
            with mock.patch.object(self.h, name) as patched:
                with self.h:
                    self.model(x)
                patched.assert_called()

        check_method_called('forward_preprocess')
        check_method_called('forward_postprocess')

    def test_all_called(self):
        x = chainer.Variable(numpy.random.rand(2, 3).astype(numpy.float32))
        self.check_hook_methods_called(x)


testing.run_module(__name__, __file__)
