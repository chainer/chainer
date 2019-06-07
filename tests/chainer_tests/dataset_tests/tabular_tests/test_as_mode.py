import unittest

import chainer
from chainer import testing
from chainer_tests.dataset_tests.tabular_tests import dummy_dataset


@testing.parameterize(
    {'mode': tuple},
    {'mode': dict},
)
class TestAsTuple(unittest.TestCase):

    def test_as_tuple(self):
        dataset = dummy_dataset.DummyDataset(mode=self.mode)
        view = dataset.as_tuple()
        self.assertIsInstance(view, chainer.dataset.TabularDataset)
        self.assertEqual(len(view), len(dataset))
        self.assertEqual(view.keys, dataset.keys)
        self.assertEqual(view.mode, tuple)


class TestAsTupleConvert(unittest.TestCase):

    def test_as_tuple_convert(self):
        def converter(a, b, c):
            return 'converted'

        dataset = dummy_dataset.DummyDataset().with_converter(converter)
        view = dataset.as_tuple()
        self.assertEqual(view.convert(view.fetch()), 'converted')


@testing.parameterize(
    {'mode': tuple},
    {'mode': dict},
)
class TestAsDict(unittest.TestCase):

    def test_as_dict(self):
        dataset = dummy_dataset.DummyDataset(mode=self.mode)
        view = dataset.as_dict()
        self.assertIsInstance(view, chainer.dataset.TabularDataset)
        self.assertEqual(len(view), len(dataset))
        self.assertEqual(view.keys, dataset.keys)
        self.assertEqual(view.mode, dict)


class TestAsDictConvert(unittest.TestCase):

    def test_as_dict_convert(self):
        def converter(a, b, c):
            return 'converted'

        dataset = dummy_dataset.DummyDataset().with_converter(converter)
        view = dataset.as_dict()
        self.assertEqual(view.convert(view.fetch()), 'converted')


testing.run_module(__name__, __file__)
