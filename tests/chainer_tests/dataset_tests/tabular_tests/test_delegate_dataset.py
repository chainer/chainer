import unittest

import numpy as np

import chainer
from chainer import testing
from chainer.dataset import tabular


from chainer_tests.dataset_tests.tabular_tests import dummy_dataset


@testing.parameterize(*testing.product({
    'mode': [tuple, dict, None],
    'overwrite_mode': [tuple, dict, None],
}))
class TestDelegateDataset(unittest.TestCase):

    def test_delegate_dataset(self):
        base_dataset = dummy_dataset.DummyDataset(mode=self.mode)
        dataset = tabular.DelegateDataset(base_dataset)

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), len(base_dataset))
        self.assertEqual(dataset.keys, base_dataset.keys)
        self.assertEqual(dataset.mode, base_dataset.mode)
        self.assertEqual(dataset.get_example(3), base_dataset.get_example(3))

    def test_overwrite_keys(self):
        base_dataset = dummy_dataset.DummyDataset(mode=self.mode)
        dataset = tabular.DelegateDataset(base_dataset)

        dataset.keys = ('a',)

        self.assertEqual(dataset.keys, ('a',))
        self.assertEqual(
            dataset.get_example(3),
            base_dataset.slice[:, ('a',)].get_example(3))

    def test_overwrite_mode(self):
        base_dataset = dummy_dataset.DummyDataset(mode=self.mode)
        dataset = tabular.DelegateDataset(base_dataset)

        if self.mode is not None and self.overwrite_mode is None:
            with self.assertRaises(ValueError):
                dataset.mode = self.overwrite_mode
            return

        dataset.mode = self.overwrite_mode

        self.assertEqual(dataset.mode, self.overwrite_mode)
        if self.overwrite_mode is tuple:
            self.assertEqual(
                dataset.get_example(3),
                base_dataset.as_tuple().get_example(3))
        elif self.overwrite_mode is tuple:
            self.assertEqual(
                dataset.get_example(3),
                base_dataset.as_dict().get_example(3))
        elif self.overwrite_mode is None:
            self.assertEqual(
                dataset.get_example(3),
                base_dataset.slice[:, 0].get_example(3))


testing.run_module(__name__, __file__)
