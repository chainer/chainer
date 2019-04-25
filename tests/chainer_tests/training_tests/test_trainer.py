import time
import traceback
import unittest

from chainer import testing
from chainer import training


class DummyExtension(training.extension.Extension):

    def __init__(self, test_case):
        self.is_called = False
        self.is_finalized = False
        self._test_case = test_case

    def __call__(self, trainer):
        self._test_case.assertTrue(trainer.is_initialized)
        self.is_called = True

    def finalize(self):
        self.is_finalized = True

    def initialize(self, trainer):
        trainer.is_initialized = True


class ErrorHandlingExtension(training.extension.Extension):

    def __init__(self):
        self.is_error_handled = False

    def __call__(self, trainer):
        pass

    def on_error(self, trainer, exception, tb):
        traceback.print_tb(tb)
        self.is_error_handled = True

    def finalize(self):
        pass

    def initialize(self, trainer):
        pass


class TheOnlyError(Exception):
    pass


class DummyCallableClass(object):

    def __init__(self, test_case):
        self.name = 'DummyCallableClass'
        self.is_called = False
        self.is_finalized = False
        self._test_case = test_case

    def __call__(self, trainer):
        self._test_case.assertTrue(trainer.is_initialized)
        self.is_called = True

    def finalize(self):
        self.is_finalized = True

    def initialize(self, trainer):
        trainer.is_initialized = True


class DummyClass(object):

    def __init__(self):
        self.is_touched = False

    def touch(self):
        self.is_touched = True


class TestTrainer(unittest.TestCase):

    def setUp(self):
        self.trainer = self._create_mock_trainer(10)
        self.trainer.is_initialized = False

    def _create_mock_trainer(self, iterations):
        trainer = testing.get_trainer_with_mock_updater(
            (iterations, 'iteration'))
        trainer.updater.update_core = lambda: time.sleep(0.001)
        return trainer

    def test_elapsed_time(self):
        with self.assertRaises(RuntimeError):
            self.trainer.elapsed_time

        self.trainer.run()

        self.assertGreater(self.trainer.elapsed_time, 0)

    def test_elapsed_time_serialization(self):
        self.trainer.run()
        serialized_time = self.trainer.elapsed_time

        new_trainer = self._create_mock_trainer(5)
        testing.save_and_load_npz(self.trainer, new_trainer)

        new_trainer.run()
        self.assertGreater(new_trainer.elapsed_time, serialized_time)

    def test_add_inherit_class_extension(self):
        dummy_extension = DummyExtension(self)
        self.trainer.extend(dummy_extension)
        self.trainer.run()
        self.assertTrue(dummy_extension.is_called)
        self.assertTrue(dummy_extension.is_finalized)

    def test_add_callable_class_extension(self):
        dummy_callable_class = DummyCallableClass(self)
        self.trainer.extend(dummy_callable_class)
        self.trainer.run()
        self.assertTrue(dummy_callable_class.is_called)
        self.assertTrue(dummy_callable_class.is_finalized)

    def test_add_lambda_extension(self):
        dummy_class = DummyClass()
        self.trainer.extend(lambda x: dummy_class.touch())
        self.trainer.run()
        self.assertTrue(dummy_class.is_touched)

    def test_add_make_extension(self):
        self.is_called = False

        @training.make_extension()
        def dummy_extension(trainer):
            self.is_called = True

        self.trainer.extend(dummy_extension)
        self.trainer.run()
        self.assertTrue(self.is_called)

    def test_add_make_extension_with_initializer(self):
        self.is_called = False

        def initializer(trainer):
            trainer.is_initialized = True

        @training.make_extension(initializer=initializer)
        def dummy_extension(trainer):
            self.assertTrue(trainer.is_initialized)
            self.is_called = True

        self.trainer.extend(dummy_extension)
        self.trainer.run()
        self.assertTrue(self.is_called)

    def test_add_function_extension(self):
        self.is_called = False

        def dummy_function(trainer):
            self.is_called = True

        self.trainer.extend(dummy_function)
        self.trainer.run()
        self.assertTrue(self.is_called)

    def test_add_two_extensions_default_priority(self):
        self.called_order = []

        @training.make_extension(trigger=(1, 'epoch'))
        def dummy_extension_1(trainer):
            self.called_order.append(1)

        @training.make_extension(trigger=(1, 'epoch'))
        def dummy_extension_2(trainer):
            self.called_order.append(2)

        self.trainer.extend(dummy_extension_1)
        self.trainer.extend(dummy_extension_2)
        self.trainer.run()
        self.assertEqual(self.called_order, [1, 2])

    def test_add_two_extensions_specific_priority(self):
        self.called_order = []

        @training.make_extension(trigger=(1, 'epoch'), priority=50)
        def dummy_extension_1(trainer):
            self.called_order.append(1)

        @training.make_extension(trigger=(1, 'epoch'), priority=100)
        def dummy_extension_2(trainer):
            self.called_order.append(2)

        self.trainer.extend(dummy_extension_1)
        self.trainer.extend(dummy_extension_2)
        self.trainer.run()
        self.assertEqual(self.called_order, [2, 1])

    def test_exception_handler(self):

        ext = ErrorHandlingExtension()
        self.trainer.extend(ext, trigger=(1, 'iteration'), priority=1)
        self.assertFalse(ext.is_error_handled)

        d = {}

        def exception_handler(trainer, exp, tb):
            d['called'] = True

        @training.make_extension(trigger=(1, 'iteration'), priority=100,
                                 on_error=exception_handler)
        def exception_raiser(trainer):
            raise TheOnlyError()
        self.trainer.extend(exception_raiser)

        dummy_extension = DummyExtension(self)
        self.trainer.extend(dummy_extension)

        with self.assertRaises(TheOnlyError):
            self.trainer.run()

        self.assertTrue(d['called'])
        self.assertTrue(ext.is_error_handled)
        self.assertTrue(dummy_extension.is_finalized)

    def test_exception_in_exception_handler(self):

        ext = ErrorHandlingExtension()
        self.trainer.extend(ext, trigger=(1, 'iteration'), priority=1)
        self.assertFalse(ext.is_error_handled)

        def exception_handler(trainer, exp, tb):
            raise ValueError('hogehoge from exception handler')

        @training.make_extension(trigger=(1, 'iteration'), priority=100,
                                 on_error=exception_handler)
        def exception_raiser(trainer):
            raise TheOnlyError()
        self.trainer.extend(exception_raiser)

        dummy_extension = DummyExtension(self)
        self.trainer.extend(dummy_extension)

        with self.assertRaises(TheOnlyError):
            self.trainer.run()

        self.assertTrue(ext.is_error_handled)
        self.assertTrue(dummy_extension.is_finalized)


testing.run_module(__name__, __file__)
