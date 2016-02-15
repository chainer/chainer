import os
import shutil
import tempfile
import unittest

import numpy
try:
    from PIL import Image
except ImportError:
    pass

from chainer import datasets
from chainer.datasets import image_dataset


_skip_unless_image_dataset_available = unittest.skipUnless(
    image_dataset.available, 'PIL is not available')


@_skip_unless_image_dataset_available
class TestImageDataset(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()
        try:
            self.n_images = 5
            self.arrays = [
                numpy.random.randint(0, 255, (3, 8, 8)).astype(numpy.float32)
                for _ in range(self.n_images)
            ]
            self.image_names = [os.path.join(self.root, '{}.png'.format(i))
                                for i in range(self.n_images)]
            for a, fn in zip(self.arrays, self.image_names):
                image = a.transpose(1, 2, 0).astype(numpy.uint8)
                Image.fromarray(image).save(fn)

            self.labels = numpy.array([0, 1, 2, 1, 2], dtype=numpy.int32)
        except:
            self.tearDown()
            raise

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_len(self):
        dataset = datasets.ImageDataset(self.image_names)
        self.assertEqual(len(dataset), self.n_images)

    def test_len_with_labels(self):
        dataset = datasets.ImageDataset(self.image_names, self.labels)
        self.assertEqual(len(dataset), self.n_images)

    def test_getitem(self):
        dataset = datasets.ImageDataset(self.image_names)
        for i in range(self.n_images):
            numpy.testing.assert_array_equal(dataset[i], self.arrays[i])

    def test_getitem_with_labels(self):
        dataset = datasets.ImageDataset(self.image_names, self.labels)
        for i in range(self.n_images):
            expect_array = self.arrays[i]
            expect_label = self.labels[i]
            xi = dataset[i]
            self.assertIsInstance(xi, tuple)
            self.assertEqual(len(xi), 2)
            actual_array, actual_label = xi
            numpy.testing.assert_array_equal(actual_array, expect_array)
            numpy.testing.assert_array_equal(actual_label, expect_label)

    def test_compute_mean(self):
        dataset = datasets.ImageDataset(self.image_names)
        expect = sum(self.arrays) / self.n_images
        numpy.testing.assert_allclose(dataset.compute_mean(), expect)

    def test_compute_mean_with_cache(self):
        dataset = datasets.ImageDataset(self.image_names)
        cache_path = os.path.join(self.root, 'mean')
        mean = dataset.compute_mean_with_cache(cache_path)
        numpy.testing.assert_array_equal(mean, sum(self.arrays) / self.n_images)
        cached = numpy.load(cache_path)
        numpy.testing.assert_array_equal(cached, mean)

    def setup_list_file(self, use_label):
        path = os.path.join(self.root, 'list')
        with open(path, 'w') as listfile:
            for fn, label in zip(self.image_names, self.labels):
                if use_label:
                    line = '{} {}\n'.format(fn, label)
                else:
                    line = '{}\n'.format(fn)
                listfile.write(line)
        return path

    def test_image_list_dataset_no_labels(self):
        path = self.setup_list_file(False)
        actual = datasets.ImageListDataset(path, False)
        expect = datasets.ImageDataset(self.image_names)
        self.assertEqual(len(actual), len(expect))
        for i in range(len(expect)):
            xi = actual[i]
            self.assertIsInstance(xi, numpy.ndarray)
            numpy.testing.assert_array_equal(xi, expect[i])

    def test_image_list_dataset_with_labels(self):
        path = self.setup_list_file(True)
        actual = datasets.ImageListDataset(path, True)
        expect = datasets.ImageDataset(self.image_names, self.labels)
        self.assertEqual(len(actual), len(expect))
        for i in range(len(expect)):
            xi = actual[i]
            self.assertIsInstance(xi, tuple)
            self.assertEqual(len(xi), 2)
            self.assertIsInstance(xi[0], numpy.ndarray)
            self.assertIsInstance(xi[1], numpy.int32)
            yi = expect[i]
            numpy.testing.assert_array_equal(xi[0], yi[0])
            numpy.testing.assert_array_equal(xi[1], yi[1])
