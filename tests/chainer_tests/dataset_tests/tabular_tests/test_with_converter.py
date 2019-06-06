import unittest

import chainer
from chainer import testing
from chainer_tests.dataset_tests.tabular_tests import dummy_dataset


@testing.parameterize(
    {'mode': tuple},
    {'mode': dict},
)
class TestWithConverter(unittest.TestCase):

    def test_with_converter(self):
        def converter(data):
            self.assertEqual(data, 'input')
            return 'converted'

        dataset = dummy_dataset.DummyDataset(mode=self.mode)
        view = dataset.with_converter(converter)
        self.assertIsInstance(view, chainer.dataset.TabularDataset)
        self.assertEqual(len(view), len(dataset))
        self.assertEqual(view.keys, dataset.keys)
        self.assertEqual(view.mode, dataset.mode)
        self.assertEqual(view.convert('input'), 'converted')


testing.run_module(__name__, __file__)
