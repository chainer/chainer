import unittest

import chainer
from chainer import testing
from chainer_tests.dataset_tests.tabular_tests import dummy_dataset


@testing.parameterize(
    {'mode': tuple},
    {'mode': dict},
    {'mode': None},
)
class TestAstuple(unittest.TestCase):

    def test_astuple(self):
        dataset = dummy_dataset.DummyDataset(mode=self.mode, convert=True)
        view = dataset.astuple()
        self.assertIsInstance(view, chainer.dataset.TabularDataset)
        self.assertEqual(len(view), len(dataset))
        self.assertEqual(view.keys, dataset.keys)
        self.assertEqual(view.mode, tuple)
        self.assertEqual(
            view.get_examples(None, None), dataset.get_examples(None, None))
        self.assertEqual(view.convert(view.fetch()), 'converted')


@testing.parameterize(
    {'mode': tuple},
    {'mode': dict},
    {'mode': None},
)
class TestAsdict(unittest.TestCase):

    def test_asdict(self):
        dataset = dummy_dataset.DummyDataset(mode=self.mode, convert=True)
        view = dataset.asdict()
        self.assertIsInstance(view, chainer.dataset.TabularDataset)
        self.assertEqual(len(view), len(dataset))
        self.assertEqual(view.keys, dataset.keys)
        self.assertEqual(view.mode, dict)
        self.assertEqual(
            view.get_examples(None, None), dataset.get_examples(None, None))
        self.assertEqual(view.convert(view.fetch()), 'converted')


testing.run_module(__name__, __file__)
