import unittest

from chainer.datasets.sliceable import SliceableDataset
from chainer.datasets.sliceable import TransformDataset
from chainer import testing


class SampleDataset(SliceableDataset):

    def __len__(self):
        return 10

    @property
    def keys(self):
        return ('item0', 'item1', 'item2')

    def get_example_by_keys(self, i, keys):
        return tuple('{:s}({:d})'.format(key, i) for key in keys)


class TestTransformDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = SampleDataset()

    def test_transform(self):
        def func(in_data):
            item0, item1, item2 = in_data
            return 'transformed_' + item0, 'transformed_' + item2

        dataset = TransformDataset(self.dataset, ('item0', 'item2'), func)
        self.assertIsInstance(dataset, SliceableDataset)
        self.assertEqual(len(dataset), len(self.dataset))
        self.assertEqual(dataset.keys, ('item0', 'item2'))
        self.assertEqual(
            dataset[3], ('transformed_item0(3)', 'transformed_item2(3)'))


testing.run_module(__name__, __file__)
