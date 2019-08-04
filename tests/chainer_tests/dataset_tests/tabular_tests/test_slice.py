import unittest
import warnings

import numpy as np
import six

import chainer
from chainer import testing
from chainer_tests.dataset_tests.tabular_tests import dummy_dataset


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

        # To reduce the number of tests,
        # drop combinations of indices and keys.
        # (check only `slice[indices]` and `slice[:, keys]`)
        if not (param['indices'] == slice(None) and
                param['get_examples_indices'] is None) and \
           not (param['keys'] is None and
                param['get_examples_key_indices'] is None):
            continue

        yield param


@testing.parameterize(*_filter_params(testing.product_dict(
    testing.product_dict(
        [{'mode': tuple}, {'mode': dict}],
        [
            {'keys': None, 'expected_keys': ('a', 'b', 'c')},
            {'keys': 1, 'expected_keys': ('b',)},
            {'keys': (1,), 'expected_keys': ('b',)},
            {'keys': 3, 'key_exception': IndexError},
            {'keys': (3,), 'key_exception': IndexError},
            {'keys': 'c', 'expected_keys': ('c',)},
            {'keys': ('c',), 'expected_keys': ('c',)},
            {'keys': 'd', 'key_exception': KeyError},
            {'keys': ('d',), 'key_exception': KeyError},
            {'keys': (-1, 'a'), 'expected_keys': ('c', 'a')},
            {'keys': (), 'expected_keys': ()},
        ],
    ) +
    testing.product_dict(
        [{'mode': None}],
        [
            {'keys': None, 'expected_keys': ('a',)},
            {'keys': 0, 'expected_keys': ('a',)},
            {'keys': (0,), 'expected_keys': ('a',)},
            {'keys': 1, 'key_exception': IndexError},
            {'keys': (1,), 'key_exception': IndexError},
            {'keys': 'a', 'expected_keys': ('a',)},
            {'keys': ('a',), 'expected_keys': ('a',)},
            {'keys': 'b', 'key_exception': KeyError},
            {'keys': ('b',), 'key_exception': KeyError},
            {'keys': (), 'expected_keys': ()},
        ],
    ),
    testing.product({
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
        {'indices': [False, 3, 9, 5, True], 'expected_len': 5},
        {'indices': [], 'expected_len': 0},
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
            data = dataset.data[:, _indices_for_numpy(self.indices)]
        else:
            view = dataset.slice[self.indices, self.keys]
            if isinstance(self.keys, tuple):
                keys = self.keys
            else:
                keys = self.keys,
            key_indices = [
                {'a': 0, 'b': 1, 'c': 2}.get(key, key) for key in keys]
            data = dataset.data[key_indices][
                :, _indices_for_numpy(self.indices)]

        self.assertIsInstance(view, chainer.dataset.TabularDataset)
        self.assertEqual(len(view), self.expected_len)
        self.assertEqual(view.keys, self.expected_keys)
        if self.keys is None:
            self.assertEqual(view.mode, self.mode)
        elif isinstance(self.keys, tuple):
            self.assertEqual(view.mode, self.mode or tuple)
        else:
            self.assertEqual(view.mode, None)

        output = view.get_examples(
            self.get_examples_indices, self.get_examples_key_indices)

        if self.get_examples_indices is not None:
            data = data[:, _indices_for_numpy(self.get_examples_indices)]
        if self.get_examples_key_indices is not None:
            data = data[list(self.get_examples_key_indices)]

        for out, d in six.moves.zip_longest(output, data):
            np.testing.assert_equal(out, d)
            if self.return_array:
                self.assertIsInstance(out, np.ndarray)
            else:
                self.assertIsInstance(out, list)


# Replace list of bool with ndarray of bool
# since old numpy cannot handle list of bool.
def _indices_for_numpy(indices):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        if len(np.empty(2)[[False, True]]) == 1:
            # new numpy
            return indices

    # old numpy
    if isinstance(indices, list) and \
       len(indices) > 0 and \
       isinstance(indices[0], bool):
        return np.array(indices)
    else:
        return indices


testing.run_module(__name__, __file__)
