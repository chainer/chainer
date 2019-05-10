import numpy as np
import unittest

from chainer import testing
from chainer.dataset import TabularDataset


class DummyDataset(TabularDataset):

    data = np.array([
        [3, 1, 4, 1, 5, 9, 2, 6, 5, 3],
        [2, 7, 1, 8, 2, 8, 1, 8, 2, 8],
        [1, 4, 1, 4, 2, 1, 3, 5, 6, 2],
    ])

    def __init__(self, mode, callback):
        self._mode = mode
        self._callback = callback

    def __len__(self):
        return self.data.shape[1]

    @property
    def keys(self):
        return ('a', 'b', 'c')

    @property
    def mode(self):
        return self._mode

    def get_examples(self, indices, key_indices):
        self._callback(indices, key_indices)

        if indices is None:
            indices = slice(0, len(self), 1)
        if isinstance(indices, slice):
            indices = range(indices.start, indices.stop, indices.step)

        if key_indices is None:
            key_indices = (0, 1, 2)

        return tuple(
            list(self.data[key_index][index] for index in indices)
            for key_index in key_indices)


@testing.parameterize(*testing.product({
    'integer': [int, np.int32],
    'seq': [list, np.array],
    'mode': [tuple, dict],
}))
class TestTabularDataset(unittest.TestCase):

    def _check_getitem(self, indices, expected_indices):
        def callback(indices, key_indices):
            self.assertEqual(indices, expected_indices)
            self.assertIsNone(key_indices)

        dataset = DummyDataset(self.mode, callback)
        output = dataset[indices]

        data = dataset.data.transpose()[indices]
        if data.ndim == 1:
            if self.mode is tuple:
                expected_output = tuple(data)
            elif self.mode is dict:
                expected_output = dict(zip(('a', 'b', 'c'), data))
        else:
            if self.mode is tuple:
                expected_output = list(tuple(d) for d in data)
            elif self.mode is dict:
                expected_output = list(
                    dict(zip(('a', 'b', 'c'), d)) for d in data)
        self.assertEqual(output, expected_output)

    def test_getitem_index(self):
        self._check_getitem(self.integer(3), [3])

    def test_getitem_index_invalid(self):
        with self.assertRaises(IndexError):
            self._check_getitem(self.integer(11), None)

    def test_getitem_indices_seq(self):
        self._check_getitem(
            self.seq([self.integer(3), self.integer(1)]), [3, 1])

    def test_getitem_indices_seq_invalid(self):
        with self.assertRaises(IndexError):
            self._check_getitem(
                self.seq([self.integer(11), self.integer(1)]), None)

    def test_getitem_indices_slice(self):
        self._check_getitem(slice(3, None, -2), slice(3, -1, -2))


testing.run_module(__name__, __file__)
