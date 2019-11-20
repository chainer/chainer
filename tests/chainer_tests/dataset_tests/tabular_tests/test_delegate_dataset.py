import unittest

import chainer
from chainer.dataset import tabular
from chainer import testing

from chainer_tests.dataset_tests.tabular_tests import dummy_dataset


@testing.parameterize(
    {'mode': tuple},
    {'mode': dict},
    {'mode': None},
)
class TestDelegateDataset(unittest.TestCase):

    def test_delegate_dataset(self):
        dataset = tabular.DelegateDataset(
            dummy_dataset.DummyDataset(mode=self.mode))

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), len(dataset.dataset))
        self.assertEqual(dataset.keys, dataset.dataset.keys)
        self.assertEqual(dataset.mode, dataset.dataset.mode)
        self.assertEqual(
            dataset.get_example(3), dataset.dataset.get_example(3))


testing.run_module(__name__, __file__)
