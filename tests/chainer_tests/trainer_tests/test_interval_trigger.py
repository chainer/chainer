import unittest

import numpy

import chainer
from chainer import datasets
from chainer import trainer


class TestIntervalTrigger(unittest.TestCase):

    def setUp(self):
        # dummy trainer
        dataset = datasets.ArrayDataset('ds', numpy.array([1, 2, 3]))
        self.trainer = chainer.Trainer(
            dataset, chainer.Link(), chainer.Optimizer())

    def test_every_iteration_trigger(self):
        trigger = trainer.IntervalTrigger(1, 'iteration')
        self.trainer.t = 0
        self.assertTrue(trigger(self.trainer))
        self.trainer.t = 5
        self.assertTrue(trigger(self.trainer))
        self.trainer.t = 7
        self.assertTrue(trigger(self.trainer))

    def test_iteration_trigger(self):
        trigger = trainer.IntervalTrigger(5, 'iteration')
        self.trainer.t = 0
        self.assertTrue(trigger(self.trainer))
        self.trainer.t = 5
        self.assertTrue(trigger(self.trainer))
        self.trainer.t = 7
        self.assertFalse(trigger(self.trainer))

    def test_every_epoch_trigger(self):
        trigger = trainer.IntervalTrigger(1, 'epoch')
        self.trainer.epoch = 0
        self.trainer.new_epoch = True
        self.assertTrue(trigger(self.trainer))
        self.trainer.new_epoch = False
        self.assertFalse(trigger(self.trainer))
        self.trainer.epoch = 5
        self.trainer.new_epoch = True
        self.assertTrue(trigger(self.trainer))
        self.trainer.new_epoch = False
        self.assertFalse(trigger(self.trainer))
        self.trainer.epoch = 7
        self.trainer.new_epoch = True
        self.assertTrue(trigger(self.trainer))
        self.trainer.new_epoch = False
        self.assertFalse(trigger(self.trainer))

    def test_epoch_trigger(self):
        trigger = trainer.IntervalTrigger(5, 'epoch')
        self.trainer.epoch = 0
        self.trainer.new_epoch = True
        self.assertTrue(trigger(self.trainer))
        self.trainer.new_epoch = False
        self.assertFalse(trigger(self.trainer))
        self.trainer.epoch = 5
        self.trainer.new_epoch = True
        self.assertTrue(trigger(self.trainer))
        self.trainer.new_epoch = False
        self.assertFalse(trigger(self.trainer))
        self.trainer.epoch = 7
        self.trainer.new_epoch = True
        self.assertFalse(trigger(self.trainer))
        self.trainer.new_epoch = False
        self.assertFalse(trigger(self.trainer))
