import unittest

import numpy as np

from chainer import testing
from chainer_tests.dataset_tests.tabular_tests import dummy_dataset


@testing.parameterize(*testing.product({
    'mode': [tuple, dict, None],
    'return_array': [True, False],
}))
class TestTabularDataset(unittest.TestCase):

    def test_fetch(self):
        def callback(indices, key_indices):
            self.assertIsNone(indices)
            self.assertIsNone(key_indices)

        dataset = dummy_dataset.DummyDataset(
            mode=self.mode, return_array=self.return_array, callback=callback)
        output = dataset.fetch()

        if self.mode is tuple:
            expected = tuple(dataset.data)
        elif self.mode is dict:
            expected = dict(zip(('a', 'b', 'c'), dataset.data))
        elif self.mode is None:
            expected = dataset.data[0]
        np.testing.assert_equal(output, expected)

        if self.mode is dict:
            output = output.values()
        elif self.mode is None:
            output = output,
        for out in output:
            if self.return_array:
                self.assertIsInstance(out, np.ndarray)
            else:
                self.assertIsInstance(out, list)

    def test_get_example(self):
        def callback(indices, key_indices):
            self.assertEqual(indices, [3])
            self.assertIsNone(key_indices)

        dataset = dummy_dataset.DummyDataset(
            mode=self.mode, return_array=self.return_array, callback=callback)

        if self.mode is tuple:
            expected = tuple(dataset.data[:, 3])
        elif self.mode is dict:
            expected = dict(zip(('a', 'b', 'c'), dataset.data[:, 3]))
        elif self.mode is None:
            expected = dataset.data[0, 3]

        self.assertEqual(dataset.get_example(3), expected)

    def test_iter(self):
        dataset = dummy_dataset.DummyDataset(
            mode=self.mode, return_array=self.return_array)
        it = iter(dataset)
        for i in range(10):
            if self.mode is tuple:
                expected = tuple(dataset.data[:, i])
            elif self.mode is dict:
                expected = dict(zip(('a', 'b', 'c'), dataset.data[:, i]))
            elif self.mode is None:
                expected = dataset.data[0, i]

            self.assertEqual(next(it), expected)

        with self.assertRaises(StopIteration):
            next(it)


testing.run_module(__name__, __file__)
