import unittest

import numpy as np

import chainer
from chainer import testing
from chainer.dataset import tabular


class TestFromData(unittest.TestCase):

    def test_from_data(self):
        dataset = tabular.from_data(
            ('a', np.arange(10)),
            [3, 1, 4, 5, 9, 2, 6, 8, 7, 0],
            c=[2, 7, 1, 8, 4, 5, 9, 0, 3, 6],
            d=-np.arange(10))

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.keys), 4)
        self.assertEqual(dataset.keys[0], 'a')
        self.assertIn('c', dataset.keys[2:])
        self.assertIn('d', dataset.keys[2:])
        self.assertEqual(dataset.mode, tuple)

        output = dataset.slice[[1, 3], ('a', 1, 'c', 'd')].fetch()
        np.testing.assert_equal(output, ([1, 3], [1, 5], [7, 8], [-1, -3]))
        self.assertIsInstance(output[0], np.ndarray)
        self.assertIsInstance(output[1], list)
        self.assertIsInstance(output[2], list)
        self.assertIsInstance(output[3], np.ndarray)

    def test_from_data_args(self):
        dataset = tabular.from_data(
            ('a', np.arange(10)),
            [3, 1, 4, 5, 9, 2, 6, 8, 7, 0])

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.keys), 2)
        self.assertEqual(dataset.keys[0], 'a')
        self.assertEqual(dataset.mode, tuple)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 3], [1, 5]))
        self.assertIsInstance(output[0], np.ndarray)
        self.assertIsInstance(output[1], list)

    def test_from_data_kwargs(self):
        dataset = tabular.from_data(
            c=[2, 7, 1, 8, 4, 5, 9, 0, 3, 6],
            d=-np.arange(10))

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.keys), 2)
        self.assertIn('c', dataset.keys)
        self.assertIn('d', dataset.keys)
        self.assertEqual(dataset.mode, dict)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, {'c': [7, 8], 'd': [-1, -3]})
        self.assertIsInstance(output['c'], list)
        self.assertIsInstance(output['d'], np.ndarray)

    def test_from_data_unique_key(self):
        dataset_a = tabular.from_data(
            np.arange(10),
            [3, 1, 4, 5, 9, 2, 6, 8, 7, 0])
        dataset_b = tabular.from_data(
            [2, 7, 1, 8, 4, 5, 9, 0, 3, 6],
            -np.arange(10))
        dataset_a.join(dataset_b)


testing.run_module(__name__, __file__)
