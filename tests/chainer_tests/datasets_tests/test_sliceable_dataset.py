import unittest

from chainer.datasets import SliceableDataset
from chainer import testing


class SampleDataset(SliceableDataset):
    def __init__(self, len):
        super(SampleDataset, self).__init__()

        self.keys = ('img', 'anno0', 'anno1', 'anno2')
        self.add_getter('img', self.get_image)
        self.add_getter(('anno0', 'anno2'), self.get_anno0_anno2)
        self.add_getter('anno1', self.get_anno1)

        self.len = len
        self.count = 0

    def __len__(self):
        return self.len

    def get_image(self, i):
        self.count += 1
        return 'img({:d})'.format(i)

    def get_anno0_anno2(self, i):
        self.count += 1
        return 'anno0({:d})'.format(i), 'anno2({:d})'.format(i)

    def get_anno1(self, i):
        self.count += 1
        return 'anno1({:d})'.format(i)


class TestSliceableDataset(unittest.TestCase):

    def setUp(self):
        self.len = 10
        self.dataset = SampleDataset(self.len)

    def test_base(self):
        self.assertEqual(len(self.dataset), self.len)
        self.assertEqual(
            self.dataset[0], ('img(0)', 'anno0(0)', 'anno1(0)', 'anno2(0)'))
        self.assertEqual(self.dataset.count, 3)

    def test_slice_single(self):
        dataset = self.dataset.slice('anno0')
        self.assertEqual(len(dataset), self.len)
        self.assertEqual(dataset[1], 'anno0(1)')
        self.assertEqual(self.dataset.count, 1)

    def test_slice_single_tuple(self):
        dataset = self.dataset.slice(('anno1',))
        self.assertEqual(len(dataset), self.len)
        self.assertEqual(dataset[2], ('anno1(2)',))
        self.assertEqual(self.dataset.count, 1)

    def test_slice_multiple(self):
        dataset = self.dataset.slice(('anno0', 'anno2'))
        self.assertEqual(len(dataset), self.len)
        self.assertEqual(dataset[3], ('anno0(3)', 'anno2(3)'))
        self.assertEqual(self.dataset.count, 1)

    def test_sub(self):
        dataset = self.dataset.sub(3, 8, 2)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(
            dataset[1], ('img(5)', 'anno0(5)', 'anno1(5)', 'anno2(5)'))
        self.assertEqual(self.dataset.count, 3)

    def test_all(self):
        dataset = self.dataset.sub(3, 8, 2).slice(('anno0', 'anno1'))
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[1], ('anno0(5)', 'anno1(5)'))
        self.assertEqual(self.dataset.count, 2)


testing.run_module(__name__, __file__)
