import unittest

import numpy as np
import six

import chainer
from chainer import testing

from . import dummy_dataset


# filter out invalid combinations of params
def _filter_params(params):
    for param in params:
        if 'expected_len' in param and \
           isinstance(param['get_examples_indices'], list) and \
           any(param['expected_len'] <= index
               for index in param['get_examples_indices']):
            continue

        if 'expected_keys' in param and \
           isinstance(param['get_examples_key_indices'], tuple) and \
           any(len(param['expected_keys']) <= key_index
               for key_index in param['get_examples_key_indices']):
            continue

        yield param


@testing.parameterize(*_filter_params(testing.product_dict(
    testing.product({
        'mode': [tuple, dict],
        'return_array': [True, False],
        'integer': [int, np.int32],
    }),
    [
        {'indices': slice(None), 'expected_len': 10},
        {'indices': [3, -2], 'expected_len': 2},
        {'indices': [11, 1], 'index_exception': IndexError},
        {'indices': [i in {1, 3} for i in range(10)], 'expected_len': 2},
        {'indices': [True] * 11, 'index_exception': ValueError},
        {'indices': slice(3, None, -2), 'expected_len': 2},
        {'indices': [], 'expected_len': 0},
    ],
    [
        {'keys': None, 'expected_keys': ('a', 'b', 'c')},
        {'keys': (1,), 'expected_keys': ('b',)},
        {'keys': (3,), 'key_exception': IndexError},
        {'keys': ('c',), 'expected_keys': ('c',)},
        {'keys': ('d',), 'key_exception': KeyError},
        {'keys': (-1, 'a'), 'expected_keys': ('c', 'a')},
        {'keys': (), 'expected_keys': ()},
    ],
    testing.product({
        'get_examples_indices': [
            None, [1], [1, 0], slice(0, 2, 1), slice(1, None, -1), []],
        'get_examples_key_indices': [None, (1,), (1, 0), ()],
    }),
)))
class TestSlice(unittest.TestCase):

    def setUp(self):
        self.exception = getattr(self, 'index_exception', None) \
            or getattr(self, 'key_exception', None)

        if isinstance(self.indices, list):
            self.indices = [
                index if isinstance(index, bool) else self.integer(index)
                for index in self.indices]

    def test_slice(self):
        def callback(indices, key_indices):
            if isinstance(self.indices, list) \
                    or isinstance(self.get_examples_indices, list):
                self.assertIsInstance(indices, list)
            elif isinstance(self.indices, slice) \
                    or isinstance(self.get_examples_indices, slice):
                self.assertIsInstance(indices, slice)
            else:
                self.assertIsNone(indices)

            if self.keys is None and self.get_examples_key_indices is None:
                self.assertIsNone(key_indices)
            else:
                self.assertIsInstance(key_indices, tuple)

        dataset = dummy_dataset.DummyDataset(
            mode=self.mode, return_array=self.return_array, callback=callback)

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

        self.assertIsInstance(view, chainer.dataset.TabularDataset)
        self.assertEqual(len(view), self.expected_len)
        self.assertEqual(view.keys, self.expected_keys)
        self.assertEqual(view.mode, self.mode)

        output = view.get_examples(
            self.get_examples_indices, self.get_examples_key_indices)

        if self.get_examples_indices is not None:
            data = data[:, self.get_examples_indices]
        if self.get_examples_key_indices is not None:
            data = data[list(self.get_examples_key_indices)]

        for out, d in six.moves.zip_longest(output, data):
            np.testing.assert_equal(out, d)
            if self.return_array:
                self.assertIsInstance(out, np.ndarray)
            else:
                self.assertIsInstance(out, list)


testing.run_module(__name__, __file__)
