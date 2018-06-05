import os
import unittest

import mock
import numpy

from chainer.dataset import download
from chainer.datasets import get_svhn
from chainer.datasets import tuple_dataset
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'withlabel': [True, False],
    'scale': [1., 255.]
}))
class TestSvhn(unittest.TestCase):
    def setUp(self):
        self.root = download.get_dataset_directory(
            os.path.join('pfnet', 'chainer', 'svhn'))

    def tearDown(self):
        if hasattr(self, 'cached_files'):
            for file in self.cached_files:
                if os.path.exists(file):
                    os.remove(file)

    @attr.slow
    def test_get_svhn(self):
        self.check_retrieval_once(['train.npz', 'test.npz'], get_svhn)

    def check_retrieval_once(self, names, retrieval_func):
        self.cached_files = [os.path.join(self.root, name) for name in names]
        train, test = retrieval_func(withlabel=self.withlabel,
                                     scale=self.scale)

        for svhn_dataset in (train, test):
            if self.withlabel:
                self.assertIsInstance(svhn_dataset,
                                      tuple_dataset.TupleDataset)
                svhn_dataset = svhn_dataset._datasets[0]
            else:
                self.assertIsInstance(svhn_dataset, numpy.ndarray)

            self.assertEqual(svhn_dataset.ndim, 4)
            self.assertEqual(svhn_dataset.shape[2], svhn_dataset.shape[3])

    # test caching - call twice
    @attr.slow
    def test_get_svhn_cached(self):
        self.check_retrieval_twice(['train.npz', 'test.npz'], get_svhn)

    def check_retrieval_twice(self, names, retrieval_func):
        self.cached_files = [os.path.join(self.root, name) for name in names]
        train, test = retrieval_func(withlabel=self.withlabel,
                                     scale=self.scale)

        with mock.patch('chainer.datasets.svhn.numpy', autospec=True) as \
                mnumpy:
            train, test = retrieval_func(withlabel=self.withlabel,
                                         scale=self.scale)
        mnumpy.savez_compressed.assert_not_called()  # creator() not called
        self.assertEqual(mnumpy.load.call_count, 2)


testing.run_module(__name__, __file__)
