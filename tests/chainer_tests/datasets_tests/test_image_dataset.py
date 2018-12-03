import os
import pickle
import unittest

import numpy

from chainer import datasets
from chainer.datasets import image_dataset
from chainer import testing


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.int32],
}))
@unittest.skipUnless(image_dataset.available, 'image_dataset is not available')
class TestImageDataset(unittest.TestCase):

    def setUp(self):
        root = os.path.join(os.path.dirname(__file__), 'image_dataset')
        path = os.path.join(root, 'img.lst')
        self.dataset = datasets.ImageDataset(path, root=root, dtype=self.dtype)

    def test_len(self):
        self.assertEqual(len(self.dataset), 2)

    def test_get(self):
        img = self.dataset.get_example(0)
        self.assertEqual(img.dtype, self.dtype)
        self.assertEqual(img.shape, (4, 300, 300))

    def test_get_grey(self):
        img = self.dataset.get_example(1)
        self.assertEqual(img.dtype, self.dtype)
        self.assertEqual(img.shape, (1, 300, 300))


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.int32],
    'label_dtype': [numpy.float32, numpy.int32],
}))
@unittest.skipUnless(image_dataset.available, 'image_dataset is not available')
class TestLabeledImageDataset(unittest.TestCase):

    def setUp(self):
        root = os.path.join(os.path.dirname(__file__), 'image_dataset')
        path = os.path.join(root, 'labeled_img.lst')
        self.dataset = datasets.LabeledImageDataset(
            path, root=root, dtype=self.dtype, label_dtype=self.label_dtype)

    def test_len(self):
        self.assertEqual(len(self.dataset), 2)

    def test_get(self):
        img, label = self.dataset.get_example(0)
        self.assertEqual(img.dtype, self.dtype)
        self.assertEqual(img.shape, (4, 300, 300))

        self.assertEqual(label.dtype, self.label_dtype)
        self.assertEqual(label, 0)

    def test_get_grey(self):
        img, label = self.dataset.get_example(1)
        self.assertEqual(img.dtype, self.dtype)
        self.assertEqual(img.shape, (1, 300, 300))

        self.assertEqual(label.dtype, self.label_dtype)
        self.assertEqual(label, 1)


@unittest.skipUnless(image_dataset.available, 'image_dataset is not available')
class TestLabeledImageDatasetInvalidFormat(unittest.TestCase):

    def test_invalid_column(self):
        root = os.path.join(os.path.dirname(__file__), 'image_dataset')
        path = os.path.join(root, 'img.lst')
        with self.assertRaises(ValueError):
            datasets.LabeledImageDataset(path)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.int32],
}))
@unittest.skipUnless(image_dataset.available, 'image_dataset is not available')
class TestZippedImageDataset(unittest.TestCase):

    def setUp(self):
        root = os.path.join(os.path.dirname(__file__), 'image_dataset')
        zipfilename = os.path.join(root, 'zipped_images_1.zip')
        self.dataset = datasets.ZippedImageDataset(zipfilename,
                                                   dtype=self.dtype)

    def test_len(self):
        self.assertEqual(len(self.dataset), 2)

    def test_get(self):
        img = self.dataset.get_example(0)
        self.assertEqual(img.dtype, self.dtype)
        self.assertEqual(img.shape, (4, 300, 300))

    def test_get_grey(self):
        img = self.dataset.get_example(1)
        self.assertEqual(img.dtype, self.dtype)
        self.assertEqual(img.shape, (1, 300, 300))


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.int32],
}))
@unittest.skipUnless(image_dataset.available, 'image_dataset is not available')
class TestMultiZippedImageDataset(unittest.TestCase):

    def setUp(self):
        root = os.path.join(os.path.dirname(__file__), 'image_dataset')
        zipfilenames = [os.path.join(root, fn) for fn
                        in ('zipped_images_1.zip', 'zipped_images_2.zip')]
        self.dataset = datasets.MultiZippedImageDataset(zipfilenames,
                                                        dtype=self.dtype)

    def test_len(self):
        self.assertEqual(len(self.dataset), 5)

    def _get_check(self, ds):
        image_formats = ((4, 300, 300), (1, 300, 300), (4, 285, 1000),
                         (3, 404, 1417), (4, 404, 1417))
        for i in range(5):
            fmt = image_formats[i]
            img = ds.get_example(i)
            self.assertEqual(img.dtype, self.dtype)
            self.assertEqual(img.shape, fmt)

    def test_get(self):
        self._get_check(self.dataset)

    def test_pickle_unpickle(self):
        dss = pickle.dumps(self.dataset)
        ds = pickle.loads(dss)
        self._get_check(ds)


@testing.parameterize(*testing.product({
    'dtype': [numpy.float32, numpy.int32],
    'label_dtype': [numpy.float32, numpy.int32],
}))
@unittest.skipUnless(image_dataset.available, 'image_dataset is not available')
class TestLabeledZippedImageDataset(unittest.TestCase):
    def setUp(self):
        root = os.path.join(os.path.dirname(__file__), 'image_dataset')
        zipfilename = os.path.join(root, 'zipped_images_1.zip')
        labelfilename = os.path.join(root, 'labeled_img.lst')
        self.dataset = datasets.LabeledZippedImageDataset(
            zipfilename, labelfilename, dtype=self.dtype,
            label_dtype=self.label_dtype)

    def test_len(self):
        self.assertEqual(len(self.dataset), 2)

    def test_get(self):
        img, label = self.dataset.get_example(0)
        self.assertEqual(img.dtype, self.dtype)
        self.assertEqual(img.shape, (4, 300, 300))

        self.assertEqual(label.dtype, self.label_dtype)
        self.assertEqual(label, 0)

    def test_get_gray(self):
        img, label = self.dataset.get_example(1)
        self.assertEqual(img.dtype, self.dtype)
        self.assertEqual(img.shape, (1, 300, 300))

        self.assertEqual(label.dtype, self.label_dtype)
        self.assertEqual(label, 1)


testing.run_module(__name__, __file__)
