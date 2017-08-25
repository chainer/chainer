import unittest

import numpy

from chainer import cuda
from chainer import datasets
from chainer import testing
from chainer.testing import attr


class TestDictDataset(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.rand(3, 4)
        self.y = numpy.random.rand(3, 5)
        self.z = numpy.random.rand(4, 4)

    def check_dict_dataset(self, x, y):
        dd = datasets.DictDataset(x=x, y=y)
        self.assertEqual(len(dd), len(x))

        for i in range(len(x)):
            example = dd[i]
            self.assertIn('x', example)
            self.assertIn('y', example)

            numpy.testing.assert_array_equal(
                cuda.to_cpu(example['x']), cuda.to_cpu(x[i]))
            numpy.testing.assert_array_equal(
                cuda.to_cpu(example['y']), cuda.to_cpu(y[i]))

    def test_dict_dataset_cpu(self):
        self.check_dict_dataset(self.x, self.y)

    @attr.gpu
    def test_dict_dataset_gpu(self):
        self.check_dict_dataset(cuda.to_gpu(self.x), cuda.to_gpu(self.y))

    def test_dict_dataset_len_mismatch(self):
        with self.assertRaises(ValueError):
            datasets.DictDataset(x=self.x, z=self.z)

    def test_dict_dataset_overrun(self):
        dd = datasets.DictDataset(x=self.x, y=self.y)
        with self.assertRaises(IndexError):
            dd[3]

    def test_dict_dataset_features(self):
        dd = datasets.DictDataset(x=self.x, y=self.y)

        fy = dd.features[:, 'y']
        self.assertTrue((self.y == fy).all())
        del fy

        fx = dd.features[:, 'x']
        self.assertTrue((self.x == fx).all())
        del fx

        fx, fy = dd.features[:, ['x', 'y']]
        self.assertTrue((self.x == fx).all())
        self.assertTrue((self.y == fy).all())
        del fx, fy

        fx, fy = dd.features[:2, ['x', 'y']]
        self.assertTrue((self.x[:2] == fx).all())
        self.assertTrue((self.y[:2] == fy).all())
        del fx, fy

        fx = dd.features[[True, False, True], ['x']]
        self.assertTrue((self.x[[0, 2]] == fx).all())
        del fx

        with self.assertRaises(TypeError):
            # features key order is not guaranteed,
            # The order of returned value fx, fy is ambiguous,
            #  so slice index access is not supported
            fx, fy = dd.features[:, :]
        with self.assertRaises(TypeError):
            fx, fy = dd.features[:1]
        with self.assertRaises(IndexError):
            # 3 is out of range
            fx = dd.features[3, 'x']
        with self.assertRaises(IndexError):
            # 'w' is not in the key
            fw = dd.features[1, 'w']


testing.run_module(__name__, __file__)
