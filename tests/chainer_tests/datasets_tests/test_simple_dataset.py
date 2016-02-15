import unittest

import numpy

from chainer import datasets


class TestSimpleDataset(unittest.TestCase):

    def setUp(self):
        self.array = numpy.random.rand(3, 4)
        self.name = 'sample_set'
        self.dataset = datasets.SimpleDataset(self.name, self.array)

    def test_name(self):
        self.assertEqual(self.dataset.name, self.name)

    def test_len(self):
        self.assertEqual(len(self.dataset), len(self.array))

    def test_getitem(self):
        for i, x in enumerate(self.array):
            numpy.testing.assert_array_equal(self.dataset[i], x)


class TestSimpleDatasetLabeled(unittest.TestCase):

    def setUp(self):
        self.array = numpy.random.rand(10, 4)
        self.labels = numpy.random.randint(0, 3, (10,))
        self.name = 'labeled_sample_set'
        self.dataset = datasets.SimpleDataset(
            self.name, (self.array, self.labels))

    def test_name(self):
        self.assertEqual(self.dataset.name, self.name)

    def test_len(self):
        self.assertEqual(len(self.dataset), len(self.array))

    def test_getitem(self):
        for i, (x, y) in enumerate(zip(self.array, self.labels)):
            item = self.dataset[i]
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            numpy.testing.assert_array_equal(item[0], x)
            numpy.testing.assert_array_equal(item[1], y)

    def test_out_of_range(self):
        with self.assertRaises(IndexError):
            self.dataset[len(self.array)]


class TestSimpleDatasetInvalid(unittest.TestCase):

    def test_len_mismatch(self):
        array1 = numpy.ndarray((3, 4))
        array2 = numpy.ndarray((2, 4))
        with self.assertRaises(ValueError):
            datasets.SimpleDataset('name', (array1, array2))
