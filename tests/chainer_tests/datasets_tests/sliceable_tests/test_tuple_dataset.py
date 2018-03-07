import unittest

import numpy as np

from chainer.datasets.sliceable import SliceableDataset
from chainer.datasets.sliceable import TupleDataset
from chainer import testing


class SampleDataset(SliceableDataset):

    def __len__(self):
        return 10

    @property
    def keys(self):
        return ('item0', 'item1', 'item2')

    def get_example_by_keys(self, i, key_indices):
        return tuple(
            '{:s}({:d})'.format(self.keys[key_index], i)
            for key_index in key_indices)


class TestTupleDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = SampleDataset()

    def test_basic(self):
        dataset = TupleDataset(
            self.dataset,
            ('item3', np.arange(len(self.dataset))),
            np.arange(len(self.dataset)) * 2)
        self.assertIsInstance(dataset, SliceableDataset)
        self.assertEqual(len(dataset), len(self.dataset))
        self.assertEqual(
            dataset.keys, ('item0', 'item1', 'item2', 'item3', None))
        self.assertEqual(
            dataset[1], ('item0(1)', 'item1(1)', 'item2(1)', 1, 2))

    def test_empty(self):
        with self.assertRaises(ValueError):
            TupleDataset()

    def test_invalid_length(self):
        with self.assertRaises(ValueError):
            TupleDataset(
                self.dataset, ('item3', np.arange(5)))


testing.run_module(__name__, __file__)
