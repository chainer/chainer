import unittest

import numpy as np
import six

import chainer
from chainer import testing
from chainer_tests.dataset_tests.tabular_tests import dummy_dataset


# filter out invalid combinations of params
def _filter_params(params):
    for param in params:
        if param['out_mode'] is None and \
           isinstance(param['key_indices'], tuple) and \
           any(1 <= key_index
               for key_index in param['key_indices']):
            continue

        yield param


@testing.parameterize(*_filter_params(testing.product({
    'in_mode': [tuple, dict],
    'out_mode': [tuple, dict, None],
    'indices': [None, [1, 3], slice(None, 2)],
    'key_indices': [None, (0,), (1, 0)],
    'with_batch': [False, True],
})))
class TestTransform(unittest.TestCase):

    def test_transform(self):
        dataset = dummy_dataset.DummyDataset(
            mode=self.in_mode, return_array=True)

        def transform(*args, **kwargs):
            if self.in_mode is tuple:
                self.assertEqual(len(args), 3)
                self.assertEqual(len(kwargs), 0)
                a, b, c = args
            elif self.in_mode is dict:
                self.assertEqual(len(args), 0)
                self.assertEqual(len(kwargs), 3)
                a, b, c = kwargs['a'], kwargs['b'], kwargs['c']

            if self.with_batch:
                self.assertIsInstance(a, np.ndarray)
                self.assertIsInstance(b, np.ndarray)
                self.assertIsInstance(c, np.ndarray)
            else:
                self.assertIsInstance(a, float)
                self.assertIsInstance(b, float)
                self.assertIsInstance(c, float)

            if self.out_mode is tuple:
                return a + b, b + c
            elif self.out_mode is dict:
                return {'alpha': a + b, 'beta': b + c}
            else:
                return a + b + c

        if self.out_mode is not None:
            if self.with_batch:
                view = dataset.transform_batch(('alpha', 'beta'), transform)
            else:
                view = dataset.transform(('alpha', 'beta'), transform)
            data = np.vstack((
                dataset.data[0] + dataset.data[1],
                dataset.data[1] + dataset.data[2]))
        else:
            if self.with_batch:
                view = dataset.transform_batch('alpha', transform)
            else:
                view = dataset.transform('alpha', transform)
            data = dataset.data.sum(axis=0, keepdims=True)

        self.assertIsInstance(view, chainer.dataset.TabularDataset)
        self.assertEqual(len(view), len(dataset))

        if self.out_mode is not None:
            self.assertEqual(view.keys, ('alpha', 'beta'))
            self.assertEqual(view.mode, self.out_mode)
        else:
            self.assertEqual(view.keys, ('alpha',))
            self.assertEqual(view.mode, tuple)

        output = view.get_examples(self.indices, self.key_indices)

        if self.indices is not None:
            data = data[:, self.indices]
        if self.key_indices is not None:
            data = data[list(self.key_indices)]

        for out, d in six.moves.zip_longest(output, data):
            np.testing.assert_equal(out, d)
            if self.with_batch:
                self.assertIsInstance(out, np.ndarray)
            else:
                self.assertIsInstance(out, list)


testing.run_module(__name__, __file__)
