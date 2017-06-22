import os
import shutil
import unittest

import numpy
import mock

from chainer.dataset import download
from chainer.datasets import tuple_dataset
from chainer.testing import attr
from chainer import datasets 


class Preprocessor(object):

    def __init__(self):
        self.call_count = 0

    def __call__(self, mol_supplier):
        descriptors = numpy.zeros((10, 10), dtype=numpy.float32)
        labels = numpy.zeros((10,), dtype=numpy.int32)
        self.call_count += 1
        return descriptors, labels


class TestTox21(unittest.TestCase):

    def _check(self, dataset, with_label=True):
        if with_label:
            self.assertIs(dataset, tuple_dataset.TupleDataset)
            first_elem = dataset[0]
            self.assertEqual(len(first_elem), 2)
        else:
            self.assertIsInstance(dataset, numpy.ndarray)

    @attr.slow
    def test_get_tox21(self):
        if not datasets.tox21.available:
            return
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
    def test_get_tox21_customized_preprocessor(self):
        if not datasets.tox21.available:
            return
        p = Preprocessor()
        train, test, val = datasets.get_tox21(p)
        self.assertEqual(p.call_count, 3)
        self._check_labeled_data(train)
        self._check_labeled_data(test)
        self._check_unlabeled_data(val)


class TestTox21Cache(unittest.TestCase):
        
    def setUp(self):
        self.dataset_root = download.get_dataset_root()
        download.set_dataset_root('/tmp')
        root = download.get_dataset_directory(
            os.path.join('pfnet', 'chainer', 'tox21'))
        if os.path.exists(root):
            shutil.rmtree(root)

    def tearDown(self):
        download.set_dataset_root = self.dataset_root

    def test(self):

        with mock.patch('chainer.datasets.tox21.Chem') as m:
            datasets.get_tox21()
            self.assertEqual(m.SDMolSupplier.call_count, 0)
            datasets.get_tox21()
            self.assertEqual(m.SDMolSupplier.call_count, 3)
