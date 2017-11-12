import unittest

import numpy

from chainer import cuda
from chainer import datasets
from chainer import testing
from chainer.testing import attr, assert_allclose


def array_equiv(x0, x1):
    # type check
    if type(x0) != type(x1):
        return False

    # numpy case
    if isinstance(x0, numpy.ndarray):
        if x0.dtype == numpy.object:
            # assert all element is same
            return (x0 == x1).all()
        else:
            # assert all element is same for object type
            if x0.shape != x1.shape:
                return False
            x0_flatten = x0.ravel()
            x1_flatten = x1.ravel()
            for x0_elem, x1_elem in zip(x0_flatten, x1_flatten):
                if ~array_equiv(x0_elem, x1_elem):
                    return False
            return True
    else:
        return x0 == x1


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

    def _test_tuple_dataset_features_1d_access(self, x0, x1):
        # Test for features indexer for one dimension accessing.
        # It extracts all features of specified indices
        td = datasets.TupleDataset(x0, x1)

        # slice
        f0, f1 = td.features[:]
        # import IPython;
        # IPython.embed()

        self.assertTrue(numpy.array_equiv(x0, f0))
        self.assertTrue(numpy.array_equiv(x1, f1))
        # self.assertTrue((x0 == f0).all())
        # self.assertTrue((x1 == f1).all())
        del f0, f1

        # int
        i = 0
        f0, f1 = td.features[i]
        assert_allclose(x0[i], f0)
        self.assertTrue(array_equiv(x0[i], f0))
        self.assertTrue(array_equiv(x1[i], f1))
        # self.assertTrue((x0[i] == f0).all())
        # self.assertTrue((x1[i] == f1).all())
        del f0, f1

        # integer list
        l = [1, 2]
        f0, f1 = td.features[l]
        print(x0.shape)
        print(x0[l, :].shape)
        print(f0.shape)
        self.assertTrue(array_equiv(x0[l, :], f0))
        self.assertTrue(array_equiv(x1[l, :], f1))
        # self.assertTrue((x0[l, :] == f0).all())
        # self.assertTrue((x1[l, :] == f1).all())
        del f0, f1

        # boolean list
        bl = [True, False, True]
        f0, f1 = td.features[bl]
        self.assertTrue(array_equiv(x0[[0, 2], :], f0))
        self.assertTrue(array_equiv(x1[[0, 2], :], f1))
        # self.assertTrue((x0[[0, 2], :] == f0).all())
        # self.assertTrue((x1[[0, 2], :] == f1).all())
        del f0, f1

        # Equivalence test (these expression works same with above)
        f0, f1 = td.features[:, :]
        self.assertTrue(array_equiv(x0, f0))
        self.assertTrue(array_equiv(x1, f1))
        # self.assertTrue((x0 == f0).all())
        # self.assertTrue((x1 == f1).all())
        del f0, f1
        f0, f1 = td.features[i, :]
        self.assertTrue(array_equiv(x0[i], f0))
        self.assertTrue(array_equiv(x1[i], f1))
        # self.assertTrue((x0[i] == f0).all())
        # self.assertTrue((x1[i] == f1).all())
        del f0, f1
        f0, f1 = td.features[l, :]
        self.assertTrue(array_equiv(x0[l], f0))
        self.assertTrue(array_equiv(x1[l, :], f1))
        # self.assertTrue((x0[l] == f0).all())
        # self.assertTrue((x1[l, :] == f1).all())
        del f0, f1
        f0, f1 = td.features[bl, :]
        self.assertTrue(array_equiv(x0[[0, 2], :], f0))
        self.assertTrue(array_equiv(x1[[0, 2], :], f1))
        # self.assertTrue((x0[[0, 2], :] == f0).all())
        # self.assertTrue((x1[[0, 2], :] == f1).all())
        del f0, f1

    def _test_tuple_dataset_features_2d_access(self, x0, x1):
        # Test 2. second dimension accessing
        td = datasets.TupleDataset(x0, x1)
        # int
        f0 = td.features[:, 0]
        self.assertTrue((x0 == f0).all())
        del f0

        f1 = td.features[0, 1]
        self.assertTrue((x1[0] == f1).all())
        del f1

        # slice
        f1 = td.features[:, -1:]
        self.assertTrue((x1[:] == f1).all())
        del f1

        # integer list
        f0 = td.features[0:5, [0, ]]
        self.assertTrue((x0[0:5] == f0).all())
        del f0

        # boolean list
        f1 = td.features[0:5, [False, True]]
        self.assertTrue((x1[0:5] == f1).all())
        del f1

    def _test_tuple_dataset_features_raise_error(self, x0, x1):
        # feature length is 2 for this dataset
        td = datasets.TupleDataset(x0, x1)

        with self.assertRaises(IndexError):
            error_index = len(x0)
            # `error_index` is out of range for dataset_length
            fx = td.features[error_index, 0]
            del fx

        with self.assertRaises(IndexError):
            # 3 is out of range for feature_length
            f_unknown = td.features[1, 3]
            del f_unknown

        with self.assertRaises(ValueError):
            # boolean flag length does not match with dataset_length
            bl = [True] * len(x0) + [True, ]
            f0, f1 = td.features[:, bl]
            del bl, f0, f1

        with self.assertRaises(ValueError):
            # boolean flag length does not match with feature_length
            bl = [True, True, True]
            f0, f1 = td.features[:, bl]
            del bl, f0, f1

    def test_tuple_dataset_features(self):
        # 1. Test for numpy array dataset
        self._test_tuple_dataset_features_1d_access(self.x0, self.x1)
        self._test_tuple_dataset_features_2d_access(self.x0, self.x1)
        self._test_tuple_dataset_features_raise_error(self.x0, self.x1)

        # 2. Test for object type dataset,
        # where the dataset may contain different shape
        x0 = numpy.array([numpy.array([0]),
                          numpy.array([1, 2]),
                          numpy.array([[10, 11], [20, 21]])])[:, None]
        # x1 = numpy.array(['a', 'bb', 'ccc'])
        x1 = numpy.arange(3)[:, None]
        self._test_tuple_dataset_features_1d_access(x0, x1)
        self._test_tuple_dataset_features_2d_access(x0, x1)
        self._test_tuple_dataset_features_raise_error(x0, x1)


testing.run_module(__name__, __file__)
