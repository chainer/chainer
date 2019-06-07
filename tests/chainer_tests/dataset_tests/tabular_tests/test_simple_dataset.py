import unittest

import numpy as np

import chainer
from chainer import testing
from chainer.dataset import tabular


class DatasetDataOnly(tabular.SimpleDataset):

    def __init__(self):
        super().__init__()
        self.add_column('a', np.arange(10))
        self.add_column('b', [3, 1, 4, 5, 9, 2, 6, 8, 7, 0])


class DatasetDataOnlyWithWrongLen(tabular.SimpleDataset):

    def __init__(self):
        super().__init__()
        self.add_column('a', np.arange(10))
        self.add_column('b', [3, 1, 4, 5, 9, 2, 6, 8, 7, 0])

    def __len__(self):
        return 12


class DatasetCallableOnly(tabular.SimpleDataset):

    def __init__(self):
        super().__init__()
        self.add_column('a', self.get_a)
        self.add_column(('b', 'c'), self.get_bc)
        self.add_column(('d', 'e'), self.get_de)

    def __len__(self):
        return 10

    def get_a(self, i):
        return 'a[{}]'.format(i)

    def get_bc(self, i):
        return 'b[{}]'.format(i), 'c[{}]'.format(i)

    def get_de(self, i):
        return {'d': 'd[{}]'.format(i), 'e': 'e[{}]'.format(i)}


class DatasetCallableOnlyWithoutLen(tabular.SimpleDataset):

    def __init__(self):
        super().__init__()
        self.add_column('a', self.get_a)
        self.add_column(('b', 'c'), self.get_bc)
        self.add_column(('d', 'e'), self.get_de)

    def get_a(self, i):
        return 'a[{}]'.format(i)

    def get_bc(self, i):
        return 'b[{}]'.format(i), 'c[{}]'.format(i)

    def get_de(self, i):
        return {'d': 'd[{}]'.format(i), 'e': 'e[{}]'.format(i)}


class DatasetMixed(tabular.SimpleDataset):

    def __init__(self):
        super().__init__()
        self.add_column('a', np.arange(10))
        self.add_column('b', self.get_b)
        self.add_column('c', [3, 1, 4, 5, 9, 2, 6, 8, 7, 0])
        self.add_column(('d', 'e'), self.get_de)

    def get_b(self, i):
        return 'b[{}]'.format(i)

    def get_de(self, i):
        return {'d': 'd[{}]'.format(i), 'e': 'e[{}]'.format(i)}


class TestSimpleDataset(unittest.TestCase):

    def test_data_only(self):
        dataset = DatasetDataOnly()

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset.keys, ('a', 'b'))
        self.assertEqual(dataset.mode, tuple)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 3], [1, 5]))
        self.assertIsInstance(output[0], np.ndarray)
        self.assertIsInstance(output[1], list)

    def test_data_only_with_wrong_len(self):
        dataset = DatasetDataOnlyWithWrongLen()

        with self.assertRaises(ValueError):
            dataset.keys

    def test_callable_only(self):
        dataset = DatasetCallableOnly()

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset.keys, ('a', 'b', 'c', 'd', 'e'))
        self.assertEqual(dataset.mode, tuple)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, (
            ['a[1]', 'a[3]'],
            ['b[1]', 'b[3]'],
            ['c[1]', 'c[3]'],
            ['d[1]', 'd[3]'],
            ['e[1]', 'e[3]']))
        for out in output:
            self.assertIsInstance(out, list)

    def test_callable_only_without_len(self):
        dataset = DatasetCallableOnlyWithoutLen()

        with self.assertRaises(NotImplementedError):
            len(dataset)
        with self.assertRaises(NotImplementedError):
            dataset.keys

    def test_mixed(self):
        dataset = DatasetMixed()

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset.keys, ('a', 'b', 'c', 'd', 'e'))
        self.assertEqual(dataset.mode, tuple)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, (
            [1, 3],
            ['b[1]', 'b[3]'],
            [1, 5],
            ['d[1]', 'd[3]'],
            ['e[1]', 'e[3]']))
        self.assertIsInstance(output[0], np.ndarray)
        self.assertIsInstance(output[1], list)
        self.assertIsInstance(output[2], list)
        self.assertIsInstance(output[3], list)
        self.assertIsInstance(output[4], list)


testing.run_module(__name__, __file__)
