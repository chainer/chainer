import unittest

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


class TestSliceableDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = SampleDataset()

    def test_base(self):
        self.assertEqual(
            self.dataset[0], ('item0(0)', 'item1(0)', 'item2(0)'))

    def test_slice_keys_single(self):
        dataset = self.dataset.slice[:, 'item0']
        self.assertIsInstance(dataset, SliceableDataset)
        self.assertEqual(len(dataset), len(self.dataset))
        self.assertEqual(dataset.keys, 'item0')
        self.assertEqual(dataset[1], 'item0(1)')

    def test_slice_keys_single_tuple(self):
        dataset = self.dataset.slice[:, ('item1',)]
        self.assertIsInstance(dataset, SliceableDataset)
        self.assertEqual(len(dataset), len(self.dataset))
        self.assertEqual(dataset.keys, ('item1',))
        self.assertEqual(dataset[2], ('item1(2)',))

    def test_slice_keys_multiple(self):
        dataset = self.dataset.slice[:, ('item0', 'item2')]
        self.assertIsInstance(dataset, SliceableDataset)
        self.assertEqual(len(dataset), len(self.dataset))
        self.assertEqual(dataset.keys, ('item0', 'item2'))
        self.assertEqual(dataset[3], ('item0(3)', 'item2(3)'))

    def test_slice_keys_invalid(self):
        with self.assertRaises(KeyError):
            self.dataset.slice[:, 'invalid']

    def test_slice_index_slice(self):
        dataset = self.dataset.slice[3:8:2]
        self.assertIsInstance(dataset, SliceableDataset)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.keys, self.dataset.keys)
        self.assertEqual(
            dataset[1], ('item0(5)', 'item1(5)', 'item2(5)'))

    def test_slice_index_list(self):
        dataset = self.dataset.slice[[2, 1, 5]]
        self.assertIsInstance(dataset, SliceableDataset)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.keys, self.dataset.keys)
        self.assertEqual(
            dataset[0], ('item0(2)', 'item1(2)', 'item2(2)'))


testing.run_module(__name__, __file__)
