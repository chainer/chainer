import numpy as np
import unittest

from chainer import testing
from chainer.dataset import TabularDataset

from .test_tabular_dataset import DummyDataset


@testing.parameterize(*testing.product_dict(
    testing.product({
        'mode_a': [tuple, dict],
        'mode_b': [tuple, dict],
        'return_array': [True, False],
    }),
    [
        {'key_indices': None,
         'expected_key_indices_a': (0, 1, 2),
         'expected_key_indices_b': (0, 1)},
        {'key_indices': (0, 4, 1),
         'expected_key_indices_a': (0, 1),
         'expected_key_indices_b': (1,)},
        {'key_indices': (0, 2),
         'expected_key_indices_a': (0, 2)},
    ],
))
class TestJoin(unittest.TestCase):

    def test_join(self):
        def callback_a(indices, key_indices):
            self.assertIsNone(indices)
            self.assertEqual(key_indices, self.expected_key_indices_a)

        dataset_a = DummyDataset(
            mode=self.mode_a,
            return_array=self.return_array, callback=callback_a)

        def callback_b(indices, key_indices):
            self.assertIsNone(indices)
            self.assertEqual(key_indices, self.expected_key_indices_b)

        dataset_b = DummyDataset(
            keys=('d', 'e'), mode=self.mode_b,
            return_array=self.return_array, callback=callback_b)

        view = dataset_a.join(dataset_b)
        self.assertIsInstance(view, TabularDataset)
        self.assertEqual(len(view), len(dataset_a))
        self.assertEqual(view.keys, dataset_a.keys + dataset_b.keys)
        self.assertEqual(view.mode, dataset_a.mode)

        data = np.vstack((dataset_a.data, dataset_b.data))
        if self.key_indices is not None:
            data = data[list(self.key_indices)]

        output = view.get_examples(None, self.key_indices)
        np.testing.assert_equal(output, data)
        for out in output:
            if self.return_array:
                self.assertIsInstance(out, np.ndarray)
            else:
                self.assertIsInstance(out, list)


class TestJoinInvalid(unittest.TestCase):

    def test_join_length(self):
        dataset_a = DummyDataset()
        dataset_b = DummyDataset(len_=5, keys=('d', 'e'))

        with self.assertRaises(ValueError):
            dataset_a.join(dataset_b)

    def test_join_conflict_key(self):
        dataset_a = DummyDataset()
        dataset_b = DummyDataset(keys=('a', 'd'))

        with self.assertRaises(ValueError):
            dataset_a.join(dataset_b)


testing.run_module(__name__, __file__)
