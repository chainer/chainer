import os
import random

import numpy
try:
    from PIL import Image
    available = True
except ImportError as e:
    available = False
    _import_error = e

from chainer import dataset


class ImageDataset(dataset.Dataset):

    """Dataset of images built from a list of paths to image files.

    TODO(beam2d): document it.

    """
    name = 'ImageDataset'

    def __init__(self, paths, labels=None, root='.', dtype=numpy.float32):
        if not available:
            raise ImportError('PIL cannot be loaded. Install pillow!\n'
                              'The actual error:\n' + str(_import_error))
        self._paths = paths
        if labels is None:
            self._labels = None
        else:
            if len(paths) != len(labels):
                raise ValueError('number of paths and labels mismatched')
            self._labels = numpy.asarray(labels, dtype=numpy.int32)
        self._root = root
        self._dtype = dtype

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, i):
        path = os.path.join(self._root, self._paths[i])
        with Image.open(path) as f:
            image = numpy.asarray(f).astype(self._dtype)
        image = image.transpose(2, 0, 1)

        if self._labels is None:
            return image
        else:
            return image, self._labels[i]

    def compute_mean(self):
        accum = 0
        for tup in self:
            image = tup if self._labels is None else tup[0]
            accum += image
        return (accum / len(self)).astype(self._dtype)

    def compute_mean_with_cache(self, cache_path):
        if os.path.exists(cache_path):
            return numpy.load(cache_path)
        mean = self.compute_mean()
        with open(cache_path, 'wb') as f:
            numpy.save(f, mean)
        return mean


class ImageListDataset(ImageDataset):

    """Dataset of images built from a image list file.

    TODO(beam2d): document it.

    """
    name = 'ImageListDataset'

    def __init__(self, path, withlabel=True, root='.', dtype=numpy.float32):
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
        super(ImageListDataset, self).__init__(paths, labels, root, dtype)
