import unittest

from chainer import testing
from chainer_tests.dataset_tests.tabular_tests import dummy_dataset


class TestWithConverter(unittest.TestCase):

    def test_with_converter(self):
        def converter(data):
            self.assertEqual(data, 'input')
            return 'converted'

        dataset = dummy_dataset.DummyDataset().with_converter(converter)
        self.assertEqual(dataset.convert('input'), 'converted')


testing.run_module(__name__, __file__)
