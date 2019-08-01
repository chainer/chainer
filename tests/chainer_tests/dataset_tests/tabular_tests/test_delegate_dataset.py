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
        base_dataset = dummy_dataset.DummyDataset(mode=self.mode)
        dataset = tabular.DelegateDataset(base_dataset)

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), len(base_dataset))
        self.assertEqual(dataset.keys, base_dataset.keys)
        self.assertEqual(dataset.mode, base_dataset.mode)
        self.assertEqual(dataset.get_example(3), base_dataset.get_example(3))


testing.run_module(__name__, __file__)
