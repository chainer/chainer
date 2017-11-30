import unittest

from chainer.datasets.sliceable import concatenate
from chainer.datasets.sliceable import SliceableDataset
from chainer import testing


class SampleDataset(SliceableDataset):

    def __len__(self):
        return 10

    @property
    def keys(self):
        return ('item0', 'item1', 'item2')

    def get_example_by_keys(self, i, keys):
        return tuple('{:s}({:d})'.format(key, i) for key in keys)


class TestConcatenate(unittest.TestCase):

    def setUp(self):
        self.dataset = SampleDataset()

    def test_concatenate(self):
        dataset = concatenate(self.dataset, self.dataset)
        self.assertIsInstance(dataset, SliceableDataset)
        self.assertEqual(len(dataset), len(self.dataset) * 2)
        self.assertEqual(dataset.keys, ('item0', 'item1', 'item2'))
        self.assertEqual(
            dataset[1], ('item0(1)', 'item1(1)', 'item2(1)'))
        self.assertEqual(
            dataset[11], ('item0(1)', 'item1(1)', 'item2(1)'))

    def test_concatenate_empty(self):
        with self.assertRaises(ValueError):
            concatenate()

    def test_concatente_invalid_keys(self):
        dataset0 = self.dataset.slice[:, ('item0', 'item1')]
        dataset1 = self.dataset.slice[:, ('item0', 'item2')]
        with self.assertRaises(ValueError):
            concatenate(dataset0, dataset1)


testing.run_module(__name__, __file__)
