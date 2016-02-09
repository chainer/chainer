import random

import numpy
from PIL import Image

from chainer import dataset


class ImageDataset(dataset.Dataset):

    """Dataset of images built from a list of paths to image files.

    TODO(beam2d): document it.

    """
    name = 'ImageDataset'

    def __init__(self, paths, labels=None, dtype=numpy.float32):
        if labels is not None and len(paths) != len(labels):
            raise ValueError('number of paths and labels mismatched')
        self._paths = paths
        self._labels = numpy.asarray(labels, dtype=numpy.int32)
        self._dtype = dtype
        self._preprocessors = []

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, i):
        path = self._paths[i]
        with Image.open(path) as f:
            image = numpy.asarray(f).astype(self._dtype)
        image = image.transpose(2, 0, 1)

        for preprocessor in self._preprocessors:
            image = preprocessor(image)

        if self._labels is None:
            return image
        else:
            return image, self._labels[i]

    def add_preprocessor(self, preprocessor):
        self._preprocessors.append(preprocessor)
        return self

    def compute_mean(self):
        accum = None
        count = 0
        for path in self._paths:
            with Image.open(path) as f:
                image = numpy.asarray(f)
            if accum is None:
                accum = image.astype(numpy.float64)
            else:
                accum += image
            count += 1
        return (accum / count).astype(self._dtype)


class ImageListDataset(ImageDataset):

    """Dataset of images built from a image list file.

    TODO(beam2d): document it.

    """
    name = 'ImageListDataset'

    def __init__(self, path, withlabel=True, dtype=numpy.float32):
        paths = []
        labels = [] if withlabel else None
        with open(path) as f:
            for line in f:
                if withlabel:
                    path, label = line.strip().split()
                    labels.append(int(label))
                else:
                    path = line.strip()
                paths.append(path)
        super(ImageListDataset, self).__init__(paths, labels, dtype)


def subtract_mean(mean):
    if isinstance(mean, str):
        with open(mean, 'rb') as f:
            mean = numpy.load(mean)
    if mean.ndim == 1:
        mean = mean[:, None, None]

    def preprocess(image):
        image -= mean
        return image

    return preprocess


def crop_center(cropsize):
    crop_h, crop_w = _get_cropsize(cropsize)

    def preprocess(image):
        _, h, w = image.shape
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        bottom = top + crop_h
        right = left + crop_w
        return image[:, top:bottom, left:right]

    return preprocess


def crop_random(cropsize):
    crop_h, crop_w = _get_cropsize(cropsize)

    def preprocess(image):
        _, h, w = image.shape
        top = random.randint(0, h - crop_h - 1)
        left = random.randint(0, w - crop_w - 1)
        bottom = top + crop_h
        right = left + crop_w
        return image[:, top:bottom, left:right]

    return preprocess


def scale(s):
    def preprocess(image):
        image *= s
        return image

    return preprocess


def random_flip(image):
    if random.randint(0, 1) == 0:
        return image[:, :, ::-1]
    else:
        return image


def _get_cropsize(cropsize):
    if isinstance(cropsize, tuple):
        return cropsize
    else:
        return cropsize, cropsize
