import numpy as np
import unittest

from chainer import testing
from chainer.dataset import TabularDataset

from .test_tabular_dataset import DummyDataset


@testing.parameterize(*testing.product_dict(
    testing.product({
        'mode': [tuple, dict],
        'integer': [int, np.int32],
        'seq': [list, np.array],
    }),
    [
        {'indices': slice(None), 'expected_len': 10,
         'expected_indices': slice(0, 10, 1)},
        {'indices': [3, 1], 'expected_len': 2, 'expected_indices': [3, 1]},
        {'indices': [11, 1], 'index_exception': IndexError},
        {'indices': [i in {1, 3} for i in range(10)],
         'expected_len': 2, 'expected_indices': [1, 3]},
        {'indices': [True] * 11, 'index_exception': ValueError},
        {'indices': slice(3, None, -2), 'expected_len': 2,
         'expected_indices': slice(3, -1, -2)},
    ],
    [
        {'keys': None, 'expected_keys': ('a', 'b', 'c')},
        {'keys': (1,), 'expected_keys': ('b',)},
        {'keys': (3,), 'key_exception': IndexError},
        {'keys': ('c',), 'expected_keys': ('c',)},
        {'keys': ('d',), 'key_exception': KeyError},
        {'keys': ('c', 0), 'expected_keys': ('c', 'a')},
    ],
    testing.product({
        'get_examples_key_indices': [None, [1], [1, 0]],
    }),
))
class TestSlice(unittest.TestCase):

    def setUp(self):
        self.exception = getattr(self, 'index_exception', None) \
            or getattr(self, 'key_exception', None)

        if isinstance(self.indices, list):
            self.indices = self.seq(
                [index if isinstance(index, bool) else self.integer(index)
                 for index in self.indices])

    def test_slice(self):
        def callback(indices, key_indices):
            self.assertEqual(indices, self.expected_indices)

            if self.keys is None:
                self.assertEqual(key_indices, self.get_examples_key_indices)
            else:
                if self.get_examples_key_indices is None:
                    keys = self.keys
                else:
                    keys = tuple(
                        self.keys[key_index]
                        for key_index in self.get_examples_key_indices)
                self.assertEqual(
                    key_indices,
                    tuple({'a': 0, 'b': 1, 'c': 2}.get(key, key)
                          for key in keys))

        dataset = DummyDataset(mode=self.mode, callback=callback)

        if self.exception is not None:
            with self.assertRaises(self.exception):
                if self.keys is None:
                    dataset.slice[self.indices]
                else:
                    dataset.slice[self.indices, self.keys]
            return

        if self.keys is None:
            view = dataset.slice[self.indices]
            data = dataset.data[:, self.indices]
        else:
            view = dataset.slice[self.indices, self.keys]
            key_indices = [
                {'a': 0, 'b': 1, 'c': 2}.get(key, key) for key in self.keys]
            data = dataset.data[key_indices][:, self.indices]

        self.assertIsInstance(view, TabularDataset)
        self.assertEqual(len(view), self.expected_len)
        self.assertEqual(view.keys, self.expected_keys)
        self.assertEqual(view.mode, self.mode)

        if self.get_examples_key_indices is not None:
            if any(key_index >= len(self.expected_keys)
                   for key_index in self.get_examples_key_indices):
                return
            data = data[self.get_examples_key_indices]

        self.assertEqual(
            view.get_examples(None, self.get_examples_key_indices),
            tuple(list(d) for d in data))


testing.run_module(__name__, __file__)
