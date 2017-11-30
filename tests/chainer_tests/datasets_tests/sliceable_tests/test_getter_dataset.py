import unittest

from chainer.datasets.sliceable import GetterDataset
from chainer import testing


class SampleDataset(GetterDataset):
    def __init__(self):
        super(SampleDataset, self).__init__()

        self.add_getter(self.get_item0, 'item0')
        self.add_getter(self.get_item1_item2, ('item1', 'item2'))
        self.add_getter(self.get_item3, 'item3')

        self.count = 0

    def __len__(self):
        return 10

    def get_item0(self, i):
        self.count += 1
        return 'item0({:d})'.format(i)

    def get_item1_item2(self, i):
        self.count += 1
        return 'item1({:d})'.format(i), 'item2({:d})'.format(i)

    def get_item3(self, i):
        self.count += 1
        return 'item3({:d})'.format(i)


class TestGetterDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = SampleDataset()

    def test_keys(self):
        self.assertEqual(
            self.dataset.keys, ('item0', 'item1', 'item2', 'item3'))

    def test_set_keys(self):
        self.dataset.keys = ('item3', 'item2', 'item1', 'item0')
        self.assertEqual(
            self.dataset.keys, ('item3', 'item2', 'item1', 'item0'))

    def test_set_keys_invalid(self):
        with self.assertRaises(KeyError):
            self.dataset.keys = ('invalid',)

    def test_get_example_by_keys(self):
        example = self.dataset.get_example_by_keys(
            1, ('item1', 'item2', 'item3'))
        self.assertEqual(example, ('item1(1)', 'item2(1)', 'item3(1)'))
        self.assertEqual(self.dataset.count, 2)


testing.run_module(__name__, __file__)
