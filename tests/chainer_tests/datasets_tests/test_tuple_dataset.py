import unittest

import numpy

from chainer import cuda
from chainer import datasets
from chainer import testing
from chainer.testing import attr


class TestTupleDataset(unittest.TestCase):

    def setUp(self):
        self.x0 = numpy.random.rand(3, 4)
        self.x1 = numpy.random.rand(3, 5)
        self.z0 = numpy.random.rand(4, 4)

    def check_tuple_dataset(self, x0, x1):
        td = datasets.TupleDataset(x0, x1)
        self.assertEqual(len(td), len(x0))

        for i in range(len(x0)):
            example = td[i]
            self.assertEqual(len(example), 2)

            numpy.testing.assert_array_equal(
                cuda.to_cpu(example[0]), cuda.to_cpu(x0[i]))
            numpy.testing.assert_array_equal(
                cuda.to_cpu(example[1]), cuda.to_cpu(x1[i]))

    def test_tuple_dataset_cpu(self):
        self.check_tuple_dataset(self.x0, self.x1)

    @attr.gpu
    def test_tuple_dataset_gpu(self):
        self.check_tuple_dataset(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1))

    def test_tuple_dataset_len_mismatch(self):
        with self.assertRaises(ValueError):
            datasets.TupleDataset(self.x0, self.z0)

    def test_tuple_dataset_overrun(self):
        td = datasets.TupleDataset(self.x0, self.x1)
        with self.assertRaises(IndexError):
            td[3]

    def test_tuple_dataset_features(self):
        td = datasets.TupleDataset(self.x0, self.x1)
        #td = datasets.TupleDataset(numpy.arange(3).astype(numpy.float32),
        #                           ['a', 'b', 'c'])


        # Test 1. one dimension accessing.
        # extracts all features of specified indices
        # slice
        f0, f1 = td.features[:]

        self.assertTrue((self.x0 == f0).all())
        self.assertTrue((self.x1 == f1).all())

        # int
        i = 0
        f0, f1 = td.features[i]
        self.assertTrue((self.x0[i] == f0).all())
        self.assertTrue((self.x1[i] == f1).all())

        # integer list
        l = [1, 2]
        f0, f1 = td.features[l]
        self.assertTrue((self.x0[l, :] == f0).all())
        self.assertTrue((self.x1[l, :] == f1).all())

        # boolean list
        bl = [True, False, True]
        f0, f1 = td.features[bl]

        #import IPython; IPython.embed()
        self.assertTrue((self.x0[[0, 2], :] == f0).all())
        self.assertTrue((self.x1[[0, 2], :] == f1).all())

        # Equivalence test
        f0, f1 = td.features[:, :]
        self.assertTrue((self.x0 == f0).all())
        self.assertTrue((self.x1 == f1).all())
        f0, f1 = td.features[i, :]
        self.assertTrue((self.x0[i] == f0).all())
        self.assertTrue((self.x1[i] == f1).all())
        f0, f1 = td.features[l, :]
        self.assertTrue((self.x0[l] == f0).all())
        self.assertTrue((self.x1[l, :] == f1).all())
        f0, f1 = td.features[bl, :]
        self.assertTrue((self.x0[[0, 2], :] == f0).all())
        self.assertTrue((self.x1[[0, 2], :] == f1).all())

        # Test 2. second dimension accessing
        # int
        f0 = td.features[:, 0]
        self.assertTrue((self.x0 == f0).all())

        f1 = td.features[0, 1]
        self.assertTrue((self.x1[0] == f1).all())

        # slice
        f1 = td.features[:, -1:]
        self.assertTrue((self.x1[:] == f1).all())

        # integer list
        f0 = td.features[0:5, [0, ]]
        self.assertTrue((self.x0[0:5] == f0).all())

        # boolean list
        f1 = td.features[0:5, [False, True]]
        self.assertTrue((self.x1[0:5] == f1).all())
        # TODO: Check string type extract, float32 type



testing.run_module(__name__, __file__)
