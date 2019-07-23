import unittest

import numpy as np

import chainer
from chainer import testing
from chainer.dataset import tabular


class TestFromData(unittest.TestCase):

    def test_unary_args(self):
        dataset = tabular.from_data(np.arange(10))

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.keys), 1)
        self.assertIsNone(dataset.mode)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, [1, 3])
        self.assertIsInstance(output, np.ndarray)

    def test_unary_args_with_key(self):
        dataset = tabular.from_data(('a', np.arange(10)))

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset.keys, ('a',))
        self.assertIsNone(dataset.mode)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, [1, 3])
        self.assertIsInstance(output, np.ndarray)

    def test_unary_kwargs(self):
        dataset = tabular.from_data(a=np.arange(10))

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset.keys, ('a',))
        self.assertIsNone(dataset.mode)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, [1, 3])
        self.assertIsInstance(output, np.ndarray)

    def test_tuple(self):
        dataset = tabular.from_data(
            np.arange(10),
            ('b', [2, 7, 1, 8, 4, 5, 9, 0, 3, 6]))

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.keys), 2)
        self.assertEqual(dataset.keys[1], 'b')
        self.assertEqual(dataset.mode, tuple)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 3], [7, 8]))
        self.assertIsInstance(output[0], np.ndarray)
        self.assertIsInstance(output[1], list)

    def test_tuple_unique(self):
        dataset_a = tabular.from_data(
            np.arange(10),
            [3, 1, 4, 5, 9, 2, 6, 8, 7, 0])
        dataset_b = tabular.from_data(
            [2, 7, 1, 8, 4, 5, 9, 0, 3, 6],
            -np.arange(10))
        self.assertFalse(set(dataset_a.keys) & set(dataset_b.keys))

    def test_dict(self):
        dataset = tabular.from_data(
            a=np.arange(10),
            b=[2, 7, 1, 8, 4, 5, 9, 0, 3, 6])

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(set(dataset.keys), {'a', 'b'})
        self.assertEqual(dataset.mode, dict)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, {'a': [1, 3], 'b': [7, 8]})
        self.assertIsInstance(output['a'], np.ndarray)
        self.assertIsInstance(output['b'], list)


testing.run_module(__name__, __file__)
