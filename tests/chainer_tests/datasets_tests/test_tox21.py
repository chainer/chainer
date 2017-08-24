import unittest

import numpy

from chainer import datasets
from chainer.datasets import tuple_dataset
from chainer.testing import attr


class Preprocessor(object):

    def __init__(self):
        self.call_count = 0

    def __call__(self, mol_supplier):
        descriptors = numpy.zeros((10, 10), dtype=numpy.float32)
        labels = numpy.zeros((10,), dtype=numpy.int32)
        self.call_count += 1
        return descriptors, labels


class TestTox21LabelNames(unittest.TestCase):

    def test_get_label_names(self):
        actual = datasets.get_tox21_label_names()
        expect = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER',
                  'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
                  'SR-HSE', 'SR-MMP', 'SR-p53']
        self.assertEqual(actual, expect)


class TestTox21(unittest.TestCase):

    def _check(self, dataset, with_label=True):
        if with_label:
            self.assertIsInstance(dataset, tuple_dataset.TupleDataset)
            first_elem = dataset[0]
            self.assertEqual(len(first_elem), 2)
        else:
            self.assertIsInstance(dataset, numpy.ndarray)

    @attr.slow
    @unittest.skipUnless(datasets.tox21.available,
                         'tox21 dataset is not available')
    def test_get_tox21(self):
        train, test, val = datasets.get_tox21()
        self._check(train)
        self._check(test)
        self._check(val, False)


class TestTox21Preprocessor(unittest.TestCase):

    def _check_labeled_data(self, dataset):
        descriptors = numpy.array([d[0] for d in dataset])
        numpy.testing.assert_array_equal(
            descriptors, numpy.zeros((10, 10), dtype=numpy.float32))
        labels = numpy.array([d[1] for d in dataset])
        numpy.testing.assert_array_equal(
            labels, numpy.zeros((10,), dtype=numpy.int32))

    def _check_unlabeled_data(self, dataset):
        numpy.testing.assert_array_equal(
            dataset, numpy.zeros((10, 10), dtype=numpy.float32))

    @attr.slow
    @unittest.skipUnless(datasets.tox21.available,
                         'tox21 dataset is not available')
    def test_get_tox21_customized_preprocessor(self):
        p = Preprocessor()
        train, test, val = datasets.get_tox21(p)
        self.assertEqual(p.call_count, 3)
        self._check_labeled_data(train)
        self._check_labeled_data(test)
        self._check_unlabeled_data(val)


class TestTox21NotAvailable(unittest.TestCase):

    @unittest.skipIf(datasets.tox21.available,
                     'tox21 dataset should not be available')
    def test_available(self):
        with self.assertRaises(RuntimeError):
            datasets.get_tox21()
