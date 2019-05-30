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
        {'key_indices': None,
         'expected_key_indices_a': None,
         'expected_key_indices_b': None},
        {'key_indices': (0, 4, 1),
         'expected_key_indices_a': (0, 1),
         'expected_key_indices_b': (1,)},
        {'key_indices': (0, 2),
         'expected_key_indices_a': (0, 2)},
        {'key_indices': (1, 2, 1),
         'expected_key_indices_a': (1, 2)},
        {'key_indices': ()},
    ],
))
class TestJoin(unittest.TestCase):

    def test_join(self):
        def callback_a(indices, key_indices):
            self.assertIsNone(indices)
            self.assertEqual(key_indices, self.expected_key_indices_a)

        dataset_a = dummy_dataset.DummyDataset(
            mode=self.mode_a,
            return_array=self.return_array, callback=callback_a)

        def callback_b(indices, key_indices):
            self.assertIsNone(indices)
            self.assertEqual(key_indices, self.expected_key_indices_b)

        dataset_b = dummy_dataset. DummyDataset(
            keys=('d', 'e'), mode=self.mode_b,
            return_array=self.return_array, callback=callback_b)

        view = dataset_a.join(dataset_b)
        self.assertIsInstance(view, chainer.dataset.TabularDataset)
        self.assertEqual(len(view), len(dataset_a))
        self.assertEqual(view.keys, dataset_a.keys + dataset_b.keys)
        self.assertEqual(view.mode, dataset_a.mode)

        output = view.get_examples(None, self.key_indices)

        data = np.vstack((dataset_a.data, dataset_b.data))
        if self.key_indices is not None:
            data = data[list(self.key_indices)]

        for out, d in six.moves.zip_longest(output, data):
            np.testing.assert_equal(out, d)
            if self.return_array:
                self.assertIsInstance(out, np.ndarray)
            else:
                self.assertIsInstance(out, list)


class TestJoinInvalid(unittest.TestCase):

    def test_join_length(self):
        dataset_a = dummy_dataset.DummyDataset()
        dataset_b = dummy_dataset.DummyDataset(size=5, keys=('d', 'e'))

        with self.assertRaises(ValueError):
            dataset_a.join(dataset_b)

    def test_join_conflict_key(self):
        dataset_a = dummy_dataset.DummyDataset()
        dataset_b = dummy_dataset.DummyDataset(keys=('a', 'd'))

        with self.assertRaises(ValueError):
            dataset_a.join(dataset_b)


testing.run_module(__name__, __file__)
