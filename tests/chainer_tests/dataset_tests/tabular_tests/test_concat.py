import operator
import unittest

import numpy as np
import six

import chainer
from chainer import testing
from chainer_tests.dataset_tests.tabular_tests import dummy_dataset


@testing.parameterize(*testing.product_dict(
    testing.product({
        'mode_a': [tuple, dict],
        'mode_b': [tuple, dict],
        'return_array': [True, False],
    }),
    [
        {'indices': None,
         'expected_indices_a': None,
         'expected_indices_b': None},
        {'indices': [3, 1, 4, 12, 14, 13, 7, 5],
         'expected_indices_a': [3, 1, 4, 7, 5],
         'expected_indices_b': [2, 4, 3]},
        {'indices': [3, 1, 4],
         'expected_indices_a': [3, 1, 4]},
        {'indices': slice(13, 6, -2),
         'expected_indices_a': slice(9, 6, -2),
         'expected_indices_b': slice(3, None, -2)},
        {'indices': slice(9, None, -2),
         'expected_indices_a': slice(9, None, -2)},
        {'indices': [1, 2, 1],
         'expected_indices_a': [1, 2, 1]},
        {'indices': []},
    ],
))
class TestConcat(unittest.TestCase):

    def test_concat(self):
        def callback_a(indices, key_indices):
            self.assertEqual(indices, self.expected_indices_a)
            self.assertIsNone(key_indices)

        dataset_a = dummy_dataset.DummyDataset(
            mode=self.mode_a,
            return_array=self.return_array, callback=callback_a)

        def callback_b(indices, key_indices):
            self.assertEqual(indices, self.expected_indices_b)
            self.assertIsNone(key_indices)

        dataset_b = dummy_dataset.DummyDataset(
            size=5, mode=self.mode_b,
            return_array=self.return_array, callback=callback_b)

        view = dataset_a.concat(dataset_b)
        self.assertIsInstance(view, chainer.dataset.TabularDataset)
        self.assertEqual(len(view), len(dataset_a) + len(dataset_b))
        self.assertEqual(view.keys, dataset_a.keys)
        self.assertEqual(view.mode, dataset_a.mode)

        output = view.get_examples(self.indices, None)

        data = np.hstack((dataset_a.data, dataset_b.data))
        if self.indices is not None:
            data = data[:, self.indices]

        for out, d in six.moves.zip_longest(output, data):
            np.testing.assert_equal(out, d)
            if self.return_array and operator.xor(
                    hasattr(self, 'expected_indices_a'),
                    hasattr(self, 'expected_indices_b')):
                self.assertIsInstance(out, np.ndarray)
            else:
                self.assertIsInstance(out, list)


class TestConcatInvalid(unittest.TestCase):

    def test_concat_key_length(self):
        dataset_a = dummy_dataset.DummyDataset()
        dataset_b = dummy_dataset.DummyDataset(keys=('a', 'b'))

        with self.assertRaises(ValueError):
            dataset_a.concat(dataset_b)

    def test_concat_key_order(self):
        dataset_a = dummy_dataset.DummyDataset()
        dataset_b = dummy_dataset.DummyDataset(keys=('b', 'a', 'c'))

        with self.assertRaises(ValueError):
            dataset_a.concat(dataset_b)


testing.run_module(__name__, __file__)
