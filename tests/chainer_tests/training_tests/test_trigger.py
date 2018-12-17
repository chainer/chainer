import unittest

import numpy

from chainer import testing
from chainer.training import trigger as trigger_module


class OnceInThreeTriggerImpl(object):
    # A trigger to trigger once in three times
    # i.e. False, False, True, False, False, True, ...

    # (this class is not a Trigger instance)

    def __init__(self, trigger_history):
        self.n = 0
        self.trigger_history = trigger_history

    def __call__(self, trainer):
        self.n += 1
        result = self.n % 3 == 0
        self.trigger_history.append(result)
        return result


@testing.parameterize(*testing.product({
    'cached_during_iteration': [True, False],
    'n_extensions': [1, 3],
}))
class TestCacheDuringIteration(unittest.TestCase):

    def create_trigger_with_decorator(self, trigger_history):
        # Make trigger by decorator

        impl = OnceInThreeTriggerImpl(trigger_history)

        @trigger_module.trigger(
            cached_during_iteration=self.cached_during_iteration)
        def decorated_trigger(trainer):
            return impl(trainer)

        return decorated_trigger

    def create_trigger_with_class(self, trigger_history):
        # Make trigger by inheritance

        impl = OnceInThreeTriggerImpl(trigger_history)

        class Trigger(trigger_module.Trigger):
            cached_during_iteration = self.cached_during_iteration

            def evaluate(self, trainer):
                return impl(trainer)

        return Trigger()

    def check_cached_during_iteration(
            self, trigger, trigger_history):
        # Register n extensions with a single trigger instance
        # and check to see if the trigger is NOT called for each extension.

        n_extensions = self.n_extensions
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
        if self.cached_during_iteration:
            expected_trigger_history = [
                i % 3 == 2 for i in range(max_iters)]
            expected_extension_epoch_details = (
                [[0.6, 1.2, 1.8]] * n_extensions)
        else:
            expected_trigger_history = [
                i % 3 == 2 for i in range(n_extensions * max_iters)]
            t = numpy.array(expected_trigger_history)
            t = t.reshape(-1, n_extensions).T
            expected_extension_epoch_details = [
                [float(j + 1) / iter_per_epoch
                 for j, value in enumerate(t[i])
                 if value]
                for i in range(n_extensions)]

        assert trigger_history == expected_trigger_history
        assert extension_epoch_details == expected_extension_epoch_details

    def test_with_decorator(self):
        history = []
        self.check_cached_during_iteration(
            self.create_trigger_with_decorator(history), history)

    def test_with_class(self):
        history = []
        self.check_cached_during_iteration(
            self.create_trigger_with_class(history), history)


testing.run_module(__name__, __file__)
