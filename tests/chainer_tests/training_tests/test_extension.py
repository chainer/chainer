import unittest

import pytest

from chainer import testing
from chainer import training


class TestExtension(unittest.TestCase):

    def test_raise_error_if_call_not_implemented(self):
        class MyExtension(training.Extension):
            pass

        ext = MyExtension()
        trainer = testing.get_trainer_with_mock_updater()
        with pytest.raises(NotImplementedError):
            ext(trainer)

    def test_default_name(self):
        class MyExtension(training.Extension):
            pass

        ext = MyExtension()
        self.assertEqual(ext.default_name, 'MyExtension')

    def test_deleted_invoke_before_training(self):
        class MyExtension(training.Extension):
            pass

        ext = MyExtension()
        with self.assertRaises(AttributeError):
            ext.invoke_before_training

    def test_make_extension(self):
        def initialize(trainer):
            pass

        @training.make_extension(trigger=(2, 'epoch'), default_name='my_ext',
                                 priority=50, initializer=initialize)
        def my_extension(trainer):
            pass

        self.assertEqual(my_extension.trigger, (2, 'epoch'))
        self.assertEqual(my_extension.default_name, 'my_ext')
        self.assertEqual(my_extension.priority, 50)
        self.assertIs(my_extension.initialize, initialize)

    def test_make_extension_default_values(self):
        @training.make_extension()
        def my_extension(trainer):
            pass

        self.assertEqual(my_extension.trigger, (1, 'iteration'))
        self.assertEqual(my_extension.default_name, 'my_extension')
        self.assertEqual(my_extension.priority, training.PRIORITY_READER)
        self.assertIsNone(my_extension.initialize)

    def test_make_extension_deleted_argument(self):
        with self.assertRaises(ValueError):
            @training.make_extension(invoke_before_training=False)
            def my_extension(_):
                pass

    def test_make_extension_unexpected_kwargs(self):
        with self.assertRaises(TypeError):
            @training.make_extension(foo=1)
            def my_extension(_):
                pass


testing.run_module(__name__, __file__)
