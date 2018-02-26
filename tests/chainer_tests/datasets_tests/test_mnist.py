import os
import unittest

import importlib
import mock
import numpy

from chainer.dataset import download
from chainer.datasets import get_fashion_mnist
from chainer.datasets import get_mnist
from chainer.datasets import tuple_dataset
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'withlabel': [True, False],
    'ndim': [1, 3],
    'scale': [1., 255.],
    'rgb_format': [True, False]
}))
class TestMnist(unittest.TestCase):

    def setUp(self):
        self.mnist_root = download.get_dataset_directory(
            os.path.join('pfnet', 'chainer', 'mnist'))
        self.fashion_mnist_root = download.get_dataset_directory(
            os.path.join('pfnet', 'chainer', 'fashion-mnist'))

    def tearDown(self):
        if (hasattr(self, 'cached_train_file') and
                os.path.exists(self.cached_train_file)):
            os.remove(self.cached_train_file)
        if (hasattr(self, 'cached_test_file') and
                os.path.exists(self.cached_test_file)):
            os.remove(self.cached_test_file)

    @attr.slow
    def test_get_mnist(self):
        self.check_retrieval_once('train.npz', 'test.npz',
                                  self.mnist_root, get_mnist)

    @attr.slow
    def test_get_fashion_mnist(self):
        self.check_retrieval_once('train.npz', 'test.npz',
                                  self.fashion_mnist_root,
                                  get_fashion_mnist)

    def check_retrieval_once(self, train_name, test_name, root,
                             retrieval_func):
        self.cached_train_file = os.path.join(root, train_name)
        self.cached_test_file = os.path.join(root, test_name)

        train, test = retrieval_func(withlabel=self.withlabel,
                                     ndim=self.ndim,
                                     scale=self.scale,
                                     rgb_format=self.rgb_format)

        for mnist_dataset in (train, test):
            if self.withlabel:
                self.assertIsInstance(mnist_dataset,
                                      tuple_dataset.TupleDataset)
                mnist_dataset = mnist_dataset._datasets[0]
            else:
                self.assertIsInstance(mnist_dataset, numpy.ndarray)

            if self.ndim == 1:
                self.assertEqual(mnist_dataset.ndim, 2)
            else:
                # self.ndim == 3
                self.assertEqual(mnist_dataset.ndim, 4)
                self.assertEqual(mnist_dataset.shape[2],
                                 mnist_dataset.shape[3])  # 32

    # test caching - call twice
    @attr.slow
    def test_get_mnist_cached(self):
        self.check_retrieval_twice('train.npz', 'test.npz',
                                   self.mnist_root,
                                   get_mnist,
                                   'chainer.datasets.mnist')

    @attr.slow
    def test_get_fashion_mnist_cached(self):
        self.check_retrieval_twice('train.npz', 'test.npz',
                                   self.fashion_mnist_root,
                                   get_fashion_mnist,
                                   'chainer.datasets.fashion_mnist')

    def check_retrieval_twice(self, train_name, test_name, root,
                              retrieval_func, package):
        self.cached_train_file = os.path.join(root, train_name)
        self.cached_test_file = os.path.join(root, test_name)
        train, test = retrieval_func(withlabel=self.withlabel,
                                     ndim=self.ndim,
                                     scale=self.scale,
                                     rgb_format=self.rgb_format)

        numpy = importlib.import_module('numpy', package=package)
        with mock.patch.object(numpy, 'savez_compressed') as savez_compressed:
            with mock.patch.object(numpy, 'load', wraps=numpy.load) as load:
                train, test = retrieval_func(withlabel=self.withlabel,
                                             ndim=self.ndim,
                                             scale=self.scale,
                                             rgb_format=self.rgb_format)
        savez_compressed.assert_not_called()  # creator() not called
        self.assertEqual(load.call_count, 2)  # for training and test


testing.run_module(__name__, __file__)
