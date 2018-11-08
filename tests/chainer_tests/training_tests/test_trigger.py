import unittest

from chainer import testing
from chainer.training import trigger as trigger_module


class OnceInThreeTrigger(object):
    # A trigger to trigger once in three times
    # i.e. False, False, True, False, False, True, ...

    def __init__(self, trigger_history):
        self.n = 0
        self.trigger_history = trigger_history

    def __call__(self, trainer):
        self.n += 1
        result = self.n % 3 == 0
        self.trigger_history.append(result)
        return result


class TestIterationAware(unittest.TestCase):

    def create_trigger_with_decorator(self, trigger_history):
        # Make iteration-aware trigger by decorator
        @trigger_module.iteration_aware()
        class DecoratedTrigger(OnceInThreeTrigger):
            pass

        return DecoratedTrigger(trigger_history)

    def create_trigger_with_decorator_without_paren(self, trigger_history):
        # Make iteration-aware trigger by decorator without parentheses `()`
        @trigger_module.iteration_aware
        class DecoratedTrigger(OnceInThreeTrigger):
            pass

        return DecoratedTrigger(trigger_history)

    def create_trigger_with_function(self, trigger_history):
        # Make iteration-aware trigger by function notation
        return trigger_module.iteration_aware(
            OnceInThreeTrigger(trigger_history))

    def check_iteration_aware(self, trigger, n_extensions, trigger_history):
        # Register n extensions with a single trigger instance
        # and check to see if the trigger is NOT called for each extension.

        max_iters = 10
        iter_per_epoch = 5

        # Create extensions
        extension_epoch_details = [[] for _ in range(n_extensions)]
        extensions = []

        def create_extension(i):
            def extension(t):
                extension_epoch_details[i].append(t.updater.epoch_detail)
            return extension

        extensions = [create_extension(i) for i in range(n_extensions)]

        # Prepare the trainer
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=(max_iters, 'iteration'),
            iter_per_epoch=iter_per_epoch)

        for i, extension in enumerate(extensions):
            trainer.extend(extension, name='ext{}'.format(i), trigger=trigger)

        # Run the trainer
        trainer.run()

        # Check
        for i in range(n_extensions):
            assert extension_epoch_details[i] == [0.6, 1.2, 1.8]
        assert len(trigger_history) == max_iters
        assert trigger_history == [
            False, False, True, False, False, True, False, False, True, False]

    def test_iteration_aware_decorator(self):
        history = []
        self.check_iteration_aware(
            self.create_trigger_with_decorator(history), 1, history)

        history = []
        self.check_iteration_aware(
            self.create_trigger_with_decorator(history), 3, history)

        history = []
        self.check_iteration_aware(
            self.create_trigger_with_decorator_without_paren(history),
            1, history)

        history = []
        self.check_iteration_aware(
            self.create_trigger_with_decorator_without_paren(history),
            3, history)

    def test_iteration_aware_function(self):
        history = []
        self.check_iteration_aware(
            self.create_trigger_with_function(history), 1, history)

        history = []
        self.check_iteration_aware(
            self.create_trigger_with_function(history), 3, history)


testing.run_module(__name__, __file__)
