import unittest

import numpy as np

import chainer
from chainer.dataset import tabular
from chainer import testing


class TestFromData(unittest.TestCase):

    def test_unary_array(self):
        dataset = tabular.from_data(np.arange(10))

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.keys), 1)
        self.assertIsNone(dataset.mode)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, [1, 3])
        self.assertIsInstance(output, np.ndarray)

    def test_unary_array_with_key(self):
        dataset = tabular.from_data(('a', np.arange(10)))

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset.keys, ('a',))
        self.assertIsNone(dataset.mode)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, [1, 3])
        self.assertIsInstance(output, np.ndarray)

    def test_unary_list(self):
        dataset = tabular.from_data([2, 7, 1, 8, 4, 5, 9, 0, 3, 6])

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.keys), 1)
        self.assertIsNone(dataset.mode)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, [7, 8])
        self.assertIsInstance(output, list)

    def test_unary_list_with_key(self):
        dataset = tabular.from_data(('a', [2, 7, 1, 8, 4, 5, 9, 0, 3, 6]))

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset.keys, ('a',))
        self.assertIsNone(dataset.mode)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, [7, 8])
        self.assertIsInstance(output, list)

    def test_unary_callable_unary(self):
        dataset = tabular.from_data(('a', lambda i: i * i), size=10)

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset.keys, ('a',))
        self.assertIsNone(dataset.mode)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, [1, 9])
        self.assertIsInstance(output, list)

    def test_unary_callable_tuple(self):
        dataset = tabular.from_data(
            (('a', 'b'), lambda i: (i * i, -i)), size=10)

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset.keys, ('a', 'b'))
        self.assertEqual(dataset.mode, tuple)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 9], [-1, -3]))
        for out in output:
            self.assertIsInstance(out, list)

    def test_unary_callable_dict(self):
        dataset = tabular.from_data(
            (('a', 'b'), lambda i: {'a': i * i, 'b': -i}), size=10)

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset.keys, ('a', 'b'))
        self.assertEqual(dataset.mode, dict)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, {'a': [1, 9], 'b': [-1, -3]})
        for out in output.values():
            self.assertIsInstance(out, list)

    def test_unary_callable_without_key(self):
        with self.assertRaises(ValueError):
            tabular.from_data(lambda i: i * i, size=10)

    def test_unary_callable_without_size(self):
        with self.assertRaises(ValueError):
            tabular.from_data(('a', lambda i: i * i))

    def test_tuple_array_list(self):
        dataset = tabular.from_data(
            (np.arange(10), [2, 7, 1, 8, 4, 5, 9, 0, 3, 6]))

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.keys), 2)
        self.assertEqual(dataset.mode, tuple)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 3], [7, 8]))
        self.assertIsInstance(output[0], np.ndarray)
        self.assertIsInstance(output[1], list)

    def test_tuple_array_with_key_list(self):
        dataset = tabular.from_data(
            (('a', np.arange(10)), [2, 7, 1, 8, 4, 5, 9, 0, 3, 6]))

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.keys), 2)
        self.assertEqual(dataset.keys[0], 'a')
        self.assertEqual(dataset.mode, tuple)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 3], [7, 8]))
        self.assertIsInstance(output[0], np.ndarray)
        self.assertIsInstance(output[1], list)

    def test_tuple_array_list_with_key(self):
        dataset = tabular.from_data(
            (np.arange(10), ('b', [2, 7, 1, 8, 4, 5, 9, 0, 3, 6])))

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.keys), 2)
        self.assertEqual(dataset.keys[1], 'b')
        self.assertEqual(dataset.mode, tuple)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 3], [7, 8]))
        self.assertIsInstance(output[0], np.ndarray)
        self.assertIsInstance(output[1], list)

    def test_tuple_array_callable_unary(self):
        dataset = tabular.from_data((np.arange(10), ('b', lambda i: i * i)))

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.keys), 2)
        self.assertEqual(dataset.keys[1], 'b')
        self.assertEqual(dataset.mode, tuple)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 3], [1, 9]))
        self.assertIsInstance(output[0], np.ndarray)
        self.assertIsInstance(output[1], list)

    def test_tuple_array_callable_tuple(self):
        dataset = tabular.from_data(
            (np.arange(10), (('b', 'c'), lambda i: (i * i, -i))))

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.keys), 3)
        self.assertEqual(dataset.keys[1:], ('b', 'c'))
        self.assertEqual(dataset.mode, tuple)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 3], [1, 9], [-1, -3]))
        self.assertIsInstance(output[0], np.ndarray)
        self.assertIsInstance(output[1], list)

    def test_tuple_array_callable_dict(self):
        dataset = tabular.from_data(
            (np.arange(10), (('b', 'c'), lambda i: {'b': i * i, 'c': -i})))

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(len(dataset.keys), 3)
        self.assertEqual(dataset.keys[1:], ('b', 'c'))
        self.assertEqual(dataset.mode, tuple)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 3], [1, 9], [-1, -3]))
        self.assertIsInstance(output[0], np.ndarray)
        self.assertIsInstance(output[1], list)

    def test_tuple_array_with_key_callable_unary(self):
        dataset = tabular.from_data(
            (('a', np.arange(10)), ('b', lambda i: i * i)))

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset.keys, ('a', 'b'))
        self.assertEqual(dataset.mode, tuple)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 3], [1, 9]))
        self.assertIsInstance(output[0], np.ndarray)
        self.assertIsInstance(output[1], list)

    def test_tuple_callable_unary_callable_unary(self):
        dataset = tabular.from_data(
            (('a', lambda i: i * i), ('b', lambda i: -i)), size=10)

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(dataset.keys, ('a', 'b'))
        self.assertEqual(dataset.mode, tuple)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, ([1, 9], [-1, -3]))
        self.assertIsInstance(output[0], list)
        self.assertIsInstance(output[1], list)

    def test_tuple_callable_unary_callable_unary_without_size(self):
        with self.assertRaises(ValueError):
            tabular.from_data((('a', lambda i: i * i), ('b', lambda i: -i)))

    def test_dict_array_list(self):
        dataset = tabular.from_data(
            {'a': np.arange(10), 'b': [2, 7, 1, 8, 4, 5, 9, 0, 3, 6]})

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(set(dataset.keys), {'a', 'b'})
        self.assertEqual(dataset.mode, dict)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, {'a': [1, 3], 'b': [7, 8]})
        self.assertIsInstance(output['a'], np.ndarray)
        self.assertIsInstance(output['b'], list)

    def test_dict_array_callable_unary(self):
        dataset = tabular.from_data({'a': np.arange(10), 'b': lambda i: i * i})

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(set(dataset.keys), {'a', 'b'})
        self.assertEqual(dataset.mode, dict)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, {'a': [1, 3], 'b': [1, 9]})
        self.assertIsInstance(output['a'], np.ndarray)
        self.assertIsInstance(output['b'], list)

    def test_dict_array_callable_tuple(self):
        dataset = tabular.from_data(
            {'a': np.arange(10), ('b', 'c'): lambda i: (i * i, -i)})

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(set(dataset.keys), {'a', 'b', 'c'})
        self.assertEqual(dataset.mode, dict)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(
            output, {'a': [1, 3], 'b': [1, 9], 'c': [-1, -3]})
        self.assertIsInstance(output['a'], np.ndarray)
        self.assertIsInstance(output['b'], list)
        self.assertIsInstance(output['c'], list)

    def test_dict_array_callable_dict(self):
        dataset = tabular.from_data(
            {'a': np.arange(10), ('b', 'c'): lambda i: {'b': i * i, 'c': -i}})

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(set(dataset.keys), {'a', 'b', 'c'})
        self.assertEqual(dataset.mode, dict)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(
            output, {'a': [1, 3], 'b': [1, 9], 'c': [-1, -3]})
        self.assertIsInstance(output['a'], np.ndarray)
        self.assertIsInstance(output['b'], list)
        self.assertIsInstance(output['c'], list)

    def test_dict_callable_unary_callable_unary(self):
        dataset = tabular.from_data(
            {'a': lambda i: i * i, 'b': lambda i: -i}, size=10)

        self.assertIsInstance(dataset, chainer.dataset.TabularDataset)
        self.assertEqual(len(dataset), 10)
        self.assertEqual(set(dataset.keys), {'a', 'b'})
        self.assertEqual(dataset.mode, dict)

        output = dataset.slice[[1, 3]].fetch()
        np.testing.assert_equal(output, {'a': [1, 9], 'b': [-1, -3]})
        self.assertIsInstance(output['a'], list)
        self.assertIsInstance(output['b'], list)

    def test_dict_callable_unary_callable_unary_without_size(self):
        with self.assertRaises(ValueError):
            tabular.from_data(({'a': lambda i: i * i, 'b': lambda i: -i}))

    def test_unique(self):
        dataset_a = tabular.from_data(np.arange(10))
        dataset_b = tabular.from_data(np.arange(10))
        self.assertNotEqual(dataset_a.keys, dataset_b.keys)


testing.run_module(__name__, __file__)
