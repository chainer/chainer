import random
import unittest

import numpy

from chainer import datasets
from chainer.datasets import cross_validation


class TestCrossValidationDataset(unittest.TestCase):

    def setUp(self):
        self.array = numpy.arange(10)
        self.base = datasets.ArrayDataset('name', self.array)
        self.order = list(range(len(self.array)))
        random.shuffle(self.order)

        self.train = [
            cross_validation.CrossValidationTrainingDataset(
                self.base, self.order, 3, i)
            for i in range(3)
        ]
        self.valid = [
            cross_validation.CrossValidationTestDataset(
                self.base, self.order, 3, i)
            for i in range(3)
        ]

    def test_len(self):
        for train, valid in zip(self.train, self.valid):
            self.assertEqual(len(train) + len(valid), len(self.base))
        self.assertEqual(sum(map(len, self.valid)), len(self.base))

    def test_getitem(self):
        expect = set(self.array)
        for train, valid in zip(self.train, self.valid):
            actual = set(list(train) + list(valid))
            self.assertEqual(actual, expect)


class TestGetCrossValidationDatasets(unittest.TestCase):

    def test_result(self):
        array = numpy.arange(10)
        base = datasets.ArrayDataset('name', array)
        n_fold = 3
        ds = datasets.get_cross_validation_datasets(base, n_fold)

        expect = set(array)
        for train, valid in ds:
            actual = set(list(train) + list(valid))
            self.assertEqual(actual, expect)
        actual = set(sum(map(list, ds[1]), []))
        self.assertEqual(actual, expect)
