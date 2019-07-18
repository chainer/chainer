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
    'in_mode': [tuple, dict, None],
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
            elif self.in_mode is None:
                self.assertEqual(len(args), 1)
                self.assertEqual(len(kwargs), 0)
                a, = args
                b, c = a, a

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
            elif self.out_mode is None:
                return a + b + c

        if self.in_mode is not None:
            a, b, c = dataset.data
        else:
            a, = dataset.data
            b, c = a, a

        if self.out_mode is not None:
            if self.with_batch:
                view = dataset.transform_batch(('alpha', 'beta'), transform)
            else:
                view = dataset.transform(('alpha', 'beta'), transform)
            data = np.vstack((a + b, b + c))
        else:
            if self.with_batch:
                view = dataset.transform_batch('alpha', transform)
            else:
                view = dataset.transform('alpha', transform)
            data = (a + b + c)[None]

        self.assertIsInstance(view, chainer.dataset.TabularDataset)
        self.assertEqual(len(view), len(dataset))

        if self.out_mode is not None:
            self.assertEqual(view.keys, ('alpha', 'beta'))
            self.assertEqual(view.mode, self.out_mode)
        else:
            self.assertEqual(view.keys, ('alpha',))
            self.assertEqual(view.mode, self.out_mode)

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


@testing.parameterize(
    {'mode': tuple},
    {'mode': dict},
    {'mode': None},
)
class TestTransformInvalid(unittest.TestCase):

    def setUp(self):
        self.count = 0

    def _transform(self, a, b, c):
        self.count += 1
        if self.count % 2 == 0:
            mode = self.mode
        else:
            if self.mode is tuple:
                mode = dict
            elif self.mode is dict:
                mode = None
            elif self.mode is None:
                mode = tuple

        if mode is tuple:
            return a,
        elif mode is dict:
            return {'a': a}
        elif mode is None:
            return a

    def test_transform_inconsistent_mode(self):
        dataset = dummy_dataset.DummyDataset()
        view = dataset.transform(('a',), self._transform)
        view.get_examples([0], None)
        with self.assertRaises(ValueError):
            view.get_examples([0], None)

    def test_transform_batch_inconsistent_mode(self):
        dataset = dummy_dataset.DummyDataset()
        view = dataset.transform_batch(('a',), self._transform)
        view.get_examples(None, None)
        with self.assertRaises(ValueError):
            view.get_examples(None, None)

    def test_transform_batch_length_changed(self):
        dataset = dummy_dataset.DummyDataset()

        def transform_batch(a, b, c):
            if self.mode is tuple:
                return a + [0],
            elif self.mode is dict:
                return {'a': a + [0]}
            elif self.mode is None:
                return a + [0]

        view = dataset.transform_batch(('a',), transform_batch)
        with self.assertRaises(ValueError):
            view.get_examples(None, None)


testing.run_module(__name__, __file__)
