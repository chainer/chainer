import unittest

import numpy as np

import chainer
from chainer import testing
from chainer.dataset.tabular import from_data


class TestFromData(unittest.TestCase):

    def test_tuple_array(self):
        dataset = from_data(
            np.arange(10),
            -np.arange(10))

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.keys), 2)
        self.assertEqual(dataset.mode, tuple)

        output = dataset.get_examples([1, 3], None)
        np.testing.assert_equal(output, ([1, 3], [-1, -3]))
        for out in output:
            self.assertIsInstance(out, np.ndarray)

    def test_tuple_list(self):
        dataset = from_data(
            [3, 1, 4, 5, 9, 2, 6, 8, 7, 0],
            [2, 7, 1, 8, 4, 5, 9, 0, 3, 6])

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.keys), 2)
        self.assertEqual(dataset.mode, tuple)

        output = dataset.get_examples([1, 3], None)
        np.testing.assert_equal(output, ([1, 5], [7, 8]))
        for out in output:
            self.assertIsInstance(out, list)

    def test_tuple_mixed(self):
        dataset = from_data(
            [3, 1, 4, 5, 9, 2, 6, 8, 7, 0],
            np.arange(10))

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.keys), 2)
        self.assertEqual(dataset.mode, tuple)

        output = dataset.get_examples([1, 3], None)
        np.testing.assert_equal(output, ([1, 5], [1, 3]))
        self.assertIsInstance(output[0], list)
        self.assertIsInstance(output[1], np.ndarray)


testing.run_module(__name__, __file__)
