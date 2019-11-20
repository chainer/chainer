import unittest

import numpy as np

import chainer
from chainer import testing
from chainer_tests.dataset_tests.tabular_tests import dummy_dataset


@testing.parameterize(
    {'mode': tuple},
    {'mode': dict},
    {'mode': None},
)
class TestWithConverter(unittest.TestCase):

    def test_with_converter(self):
        dataset = dummy_dataset.DummyDataset(mode=self.mode)

        def converter(*args, **kwargs):
            if self.mode is tuple:
                np.testing.assert_equal(args, tuple(dataset.data))
                self.assertEqual(kwargs, {})
            elif self.mode is dict:
                self.assertEqual(args, ())
                np.testing.assert_equal(
                    kwargs, dict(zip(('a', 'b', 'c'), dataset.data)))
            elif self.mode is None:
                np.testing.assert_equal(args, tuple(dataset.data))
                self.assertEqual(kwargs, {})

            return 'converted'

        view = dataset.with_converter(converter)
        self.assertIsInstance(view, chainer.dataset.TabularDataset)
        self.assertEqual(len(view), len(dataset))
        self.assertEqual(view.keys, dataset.keys)
        self.assertEqual(view.mode, dataset.mode)
        self.assertEqual(
            view.get_examples(None, None), dataset.get_examples(None, None))
        self.assertEqual(view.convert(view.fetch()), 'converted')


testing.run_module(__name__, __file__)
