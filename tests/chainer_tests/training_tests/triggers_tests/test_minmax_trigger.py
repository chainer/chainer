import tempfile
import unittest

from chainer import serializers
from chainer import testing
from chainer.training import triggers


class BestValueTriggerTester(object):
    def _test_trigger(self, trigger, key, accuracies, expected,
                      resume=None, save=None):
        trainer = testing.get_trainer_with_mock_updater(
            stop_trigger=(len(accuracies), 'iteration'),
            iter_per_epoch=self.iter_per_epoch)
        updater = trainer.updater

        def _serialize_updater(serializer):
            updater.iteration = serializer('iteration', updater.iteration)
            updater.epoch = serializer('epoch', updater.epoch)
            updater.is_new_epoch = serializer(
                'is_new_epoch', updater.is_new_epoch)
        trainer.updater.serialize = _serialize_updater

        def set_observation(t):
            t.observation = {key: accuracies[t.updater.iteration-1]}
        trainer.extend(set_observation, name='set_observation',
                       trigger=(1, 'iteration'), priority=2)

        invoked_iterations = []

        def record(t):
            invoked_iterations.append(t.updater.iteration)
        trainer.extend(record, name='record', trigger=trigger, priority=1)

        if resume is not None:
            serializers.load_npz(resume, trainer)

        trainer.run()
        self.assertEqual(invoked_iterations, expected)

        if save is not None:
            serializers.save_npz(save, trainer)

    def test_trigger(self):
        trigger = type(self).trigger_type(self.key, trigger=self.interval)
        self._test_trigger(trigger, self.key, self.accuracies, self.expected)

    def test_resumed_trigger(self):
        trigger = type(self).trigger_type(self.key, trigger=self.interval)
        with tempfile.NamedTemporaryFile() as npz:
            self._test_trigger(
                trigger, self.key, self.accuracies[:self.resume],
                self.expected_before_resume, save=npz)
            npz.flush()
            trigger = type(self).trigger_type(self.key, trigger=self.interval)
            self._test_trigger(trigger, self.key, self.accuracies,
                               self.expected_after_resume, resume=npz.name)


@testing.parameterize(
    # interval = 1 iterations
    {
        'key': 'main/accuracy',
        'iter_per_epoch': 1,
        'interval': (1, 'iteration'),
        'accuracies': [0.5, 0.5, 0.4, 0.6],
        'expected': [1, 4],
        'resume': 1,
        'expected_before_resume': [1],
        'expected_after_resume': [4]},
    # interval = 2 iterations
    {
        'key': 'main/accuracy',
        'iter_per_epoch': 1,
        'interval': (2, 'iteration'),
        'accuracies': [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
        'expected': [2, 8],
        'resume': 2,
        'expected_before_resume': [2],
        'expected_after_resume': [8]},
    # interval = 2 iterations, unaligned resume
    {
        'key': 'main/accuracy',
        'iter_per_epoch': 1,
        'interval': (2, 'iteration'),
        'accuracies': [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
        'expected': [2, 8],
        'resume': 3,
        'expected_before_resume': [2],
        'expected_after_resume': [8]},
    # interval = 1 epoch, 1 epoch = 2 iterations
    {
        'key': 'main/accuracy',
        'iter_per_epoch': 2,
        'interval': (1, 'epoch'),
        'accuracies': [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
        'expected': [2, 8],
        'resume': 2,
        'expected_before_resume': [2],
        'expected_after_resume': [8]},
    # interval = 1 epoch, 1 epoch = 2 iterations, unaligned resume
    {
        'key': 'main/accuracy',
        'iter_per_epoch': 2,
        'interval': (1, 'epoch'),
        'accuracies': [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
        'expected': [2, 8],
        'resume': 3,
        'expected_before_resume': [2],
        'expected_after_resume': [8]},
)
class TestMaxValueTrigger(unittest.TestCase, BestValueTriggerTester):
    trigger_type = triggers.MaxValueTrigger


@testing.parameterize(
    # interval = 1 iterations
    {
        'key': 'main/accuracy',
        'iter_per_epoch': 1,
        'interval': (1, 'iteration'),
        'accuracies': [0.5, 0.5, 0.4, 0.6],
        'expected': [1, 3],
        'resume': 1,
        'expected_before_resume': [1],
        'expected_after_resume': [3]},
    # interval = 2 iterations
    {
        'key': 'main/accuracy',
        'iter_per_epoch': 1,
        'interval': (2, 'iteration'),
        'accuracies': [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
        'expected': [2, 6],
        'resume': 2,
        'expected_before_resume': [2],
        'expected_after_resume': [6]},
    # interval = 2 iterations, unaligned resume
    {
        'key': 'main/accuracy',
        'iter_per_epoch': 1,
        'interval': (2, 'iteration'),
        'accuracies': [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
        'expected': [2, 6],
        'resume': 3,
        'expected_before_resume': [2],
        'expected_after_resume': [6]},
    # interval = 1 epoch, 1 epoch = 2 iterations
    {
        'key': 'main/accuracy',
        'iter_per_epoch': 2,
        'interval': (1, 'epoch'),
        'accuracies': [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
        'expected': [2, 6],
        'resume': 2,
        'expected_before_resume': [2],
        'expected_after_resume': [6]},
    # interval = 1 epoch, 1 epoch = 2 iterations, unaligned resume
    {
        'key': 'main/accuracy',
        'iter_per_epoch': 2,
        'interval': (1, 'epoch'),
        'accuracies': [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
        'expected': [2, 6],
        'resume': 3,
        'expected_before_resume': [2],
        'expected_after_resume': [6]},
)
class TestMinValueTrigger(unittest.TestCase, BestValueTriggerTester):
    trigger_type = triggers.MinValueTrigger


testing.run_module(__name__, __file__)
