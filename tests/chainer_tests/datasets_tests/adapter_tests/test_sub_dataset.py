import random
import unittest

import numpy

from chainer import datasets


class TestSubDataset(unittest.TestCase):

    def setUp(self):
        self.array = numpy.random.rand(10, 3)
        self.baseset = datasets.SimpleDataset('name', self.array)

    def test_whole_set(self):
        dataset = datasets.SubDataset(self.baseset, 0, len(self.array))
        self.assertEqual(len(dataset), len(self.array))
        for i in range(len(self.array)):
            numpy.testing.assert_array_equal(dataset[i], self.array[i])
        with self.assertRaises(IndexError):
            dataset[len(self.array)]

    def test_left_subset(self):
        dataset = datasets.SubDataset(self.baseset, 0, 5)
        self.assertEqual(len(dataset), 5)
        for i in range(5):
            numpy.testing.assert_array_equal(dataset[i], self.array[i])
        with self.assertRaises(IndexError):
            dataset[-6]
        for i in range(-5, 0):
            numpy.testing.assert_array_equal(dataset[i], self.array[5 + i])
        with self.assertRaises(IndexError):
            dataset[5]

    def test_middle_subset(self):
        dataset = datasets.SubDataset(self.baseset, 2, 8)
        self.assertEqual(len(dataset), 6)
        for i in range(6):
            numpy.testing.assert_array_equal(dataset[i], self.array[i + 2])
        with self.assertRaises(IndexError):
            dataset[6]
        for i in range(-6, 0):
            numpy.testing.assert_array_equal(dataset[i], self.array[8 + i])
        with self.assertRaises(IndexError):
            dataset[-7]

    def test_right_subset(self):
        dataset = datasets.SubDataset(self.baseset, 6, 10)
        self.assertEqual(len(dataset), 4)
        for i in range(4):
            numpy.testing.assert_array_equal(dataset[i], self.array[i + 6])
        with self.assertRaises(IndexError):
            dataset[4]
        for i in range(-4, 0):
            numpy.testing.assert_array_equal(dataset[i], self.array[10 + i])
        with self.assertRaises(IndexError):
            dataset[-5]

    def test_ordered_subset(self):
        order = list(range(10))
        random.shuffle(order)
        dataset = datasets.SubDataset(self.baseset, 2, 6, order)
        self.assertEqual(len(dataset), 4)
        for i in range(4):
            numpy.testing.assert_array_equal(dataset[i], self.array[order[i + 2]])
        with self.assertRaises(IndexError):
            dataset[4]
        for i in range(-4, 0):
            numpy.testing.assert_array_equal(dataset[i], self.array[order[6 + i]])

    def test_split_dataset(self):
        sub1, sub2 = datasets.split_dataset(self.baseset, 7)
        self.assertEqual(len(sub1), 7)
        self.assertEqual(len(sub2), len(self.array) - 7)
        for i in range(7):
            numpy.testing.assert_array_equal(sub1[i], self.array[i])
        for i in range(len(self.array) - 7):
            numpy.testing.assert_array_equal(sub2[i], self.array[i + 7])

    def test_ordered_split_dataset(self):
        order = list(range(10))
        random.shuffle(order)
        sub1, sub2 = datasets.split_dataset(self.baseset, 7, order)
        self.assertEqual(len(sub1), 7)
        self.assertEqual(len(sub2), len(self.array) - 7)
        for i in range(7):
            numpy.testing.assert_array_equal(sub1[i], self.array[order[i]])
        for i in range(len(self.array) - 7):
            numpy.testing.assert_array_equal(sub2[i], self.array[order[i + 7]])

    def test_split_dataset_random(self):
        # use scalar dataset to use set comparison
        array = numpy.arange(10)
        baseset = datasets.SimpleDataset('base', array)
        sub1, sub2 = datasets.split_dataset_random(baseset, 7)
        self.assertEqual(len(sub1), 7)
        self.assertEqual(len(sub2), len(self.array) - 7)
        self.assertEqual(set(list(sub1) + list(sub2)), set(array))
