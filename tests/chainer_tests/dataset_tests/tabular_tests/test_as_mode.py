import unittest

from chainer import testing
from chainer.dataset import TabularDataset

from .test_tabular_dataset import DummyDataset


@testing.parameterize(
    {'mode': tuple},
    {'mode': dict},
)
class TestAsTuple(unittest.TestCase):

    def test_as_tuple(self):
        dataset = DummyDataset(self.mode)
        view = dataset.as_tuple()
        self.assertIsInstance(view, TabularDataset)
        self.assertEqual(len(view), len(dataset))
        self.assertEqual(view.keys, dataset.keys)
        self.assertEqual(view.mode, tuple)


@testing.parameterize(
    {'mode': tuple},
    {'mode': dict},
)
class TestAsDict(unittest.TestCase):

    def test_as_dict(self):
        dataset = DummyDataset(self.mode)
        view = dataset.as_dict()
        self.assertIsInstance(view, TabularDataset)
        self.assertEqual(len(view), len(dataset))
        self.assertEqual(view.keys, dataset.keys)
        self.assertEqual(view.mode, dict)


testing.run_module(__name__, __file__)
