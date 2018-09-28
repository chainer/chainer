import tempfile
import unittest

import numpy

from chainer import serializers
from chainer import testing
from chainer.training import triggers


def _test_trigger(self, trigger, key, accuracies, expected,
                  resume=None, save=None):
    trainer = testing.get_trainer_with_mock_updater(
        stop_trigger=(len(accuracies), 'iteration'))

    def set_observation(t):
        t.observation = {key: accuracies[t.updater.iteration-1]}
    trainer.extend(set_observation, name='set_observation',
                   trigger=(1, 'iteration'), priority=2)

    invoked_intervals = []
    trainer.extend(lambda t: invoked_intervals.append(
        t.updater.iteration), name='test', trigger=trigger, priority=1)

    if resume is not None:
        serializers.load_npz(resume, trainer)

    trainer.run()
    self.assertEqual(invoked_intervals, expected)

    if save is not None:
        serializers.save_npz(save, trainer)


class TestMaxValueTrigger(unittest.TestCase):

    def test_max_value_trigger(self):
        key = 'main/accuracy'
        trigger = triggers.MaxValueTrigger(key, trigger=(2, 'iteration'))
        accuracies = numpy.asarray([0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
                                   dtype=numpy.float32)
        expected = [2, 8]
        _test_trigger(self, trigger, key, accuracies, expected)

    def test_resumed_trigger(self):
        key = 'main/accuracy'
        trigger = triggers.MaxValueTrigger(key, trigger=(1, 'iteration'))
        accuracies = numpy.asarray([0.5],
                                   dtype=numpy.float32)
        expected = [1]
        with tempfile.NamedTemporaryFile() as npz:
            _test_trigger(self, trigger, key, accuracies, expected, save=npz)
            npz.flush()
            trigger = triggers.MaxValueTrigger(key, trigger=(1, 'iteration'))
            accuracies = numpy.asarray([None, 0.4, 0.5, 0.6],
                                       dtype=numpy.float32)
            expected = [4]
            _test_trigger(self, trigger, key, accuracies,
                          expected, resume=npz.name)


class TestMinValueTrigger(unittest.TestCase):

    def test_min_value_trigger(self):
        key = 'main/accuracy'
        trigger = triggers.MinValueTrigger(key, trigger=(2, 'iteration'))
        accuracies = numpy.asarray([0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6],
                                   dtype=numpy.float32)
        expected = [2, 6]
        _test_trigger(self, trigger, key, accuracies, expected)

    def test_resumed_trigger(self):
        key = 'main/accuracy'
        trigger = triggers.MinValueTrigger(key, trigger=(1, 'iteration'))
        accuracies = numpy.asarray([0.5],
                                   dtype=numpy.float32)
        expected = [1]
        _test_trigger(self, trigger, key, accuracies, expected)
        with tempfile.NamedTemporaryFile() as npz:
            serializers.save_npz(npz, trigger)
            npz.flush()
            trigger = triggers.MinValueTrigger(key, trigger=(1, 'iteration'))
            serializers.load_npz(npz.name, trigger)
            accuracies = numpy.asarray([None, 0.6, 0.5, 0.4],
                                       dtype=numpy.float32)
            expected = [4]
            _test_trigger(self, trigger, key, accuracies, expected)


testing.run_module(__name__, __file__)
