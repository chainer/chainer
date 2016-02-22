import os

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

    This dataset reads an external image file on every call of the
    :meth:`__getitem__` operator. The path of the image to retrieve is given as
    a list of strings.

    If a list of labels is also given, then this dataset returns a tuple of an
    array and a label. Otherwise, it returns a single array.

    Each image is automatically converted to float32 arrays of shape
    ``channels, height, width``, where ``channels`` represents the number of
    channels in each pixel (e.g. 1 for grey-scale images, and 3 for RGB-color
    images).

    **This dataset requires the Pillow package installed.** In order to use
    this dataset, install Pillow (e.g. by using the command ``pip install
    Pillow``). Be careful to prepare appropriate libraries for image formats
    you want to use.

    Args:
        paths (list of strs): List of paths to image files. The ``i``-th file
            is used as the ``i``-th example in this dataset. Each path is
            relative to the root path.
        labels (array-like): List of labels. If this is None, then the dataset
            behaves as an unlabeled dataset.
        root (str): Root directory to retrieve images.
        dtype: Data type of resulting image arrays.

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
        """Computes the mean image of the dataset.

        This method computes the mean image and returns it. If you want to
        cache the result on the storage and reuse it, use
        :meth:`compute_mean_with_cache` instead.

        Returns:
            numpy.ndarray: Mean image of the dataset.

        """
        accum = 0
        for tup in self:
            image = tup if self._labels is None else tup[0]
            accum += image
        return (accum / len(self)).astype(self._dtype)

    def compute_mean_with_cache(self, cache_path):
        """Computes the mean image or returns cached one.

        If a cached result exists, then this method just reads the cache file
        and returns the recovered array. Otherwise, this method computes the
        mean array and saves it to the given path.

        Args:
            cache_path (str): Path to load or save the mean image.

        Returns:
            numpy.ndarray: (Possibly cached) mean image of the dataset

        """
        if os.path.exists(cache_path):
            return numpy.load(cache_path)
        mean = self.compute_mean()
        with open(cache_path, 'wb') as f:
            numpy.save(f, mean)
        return mean


class ImageListDataset(ImageDataset):

    """Dataset of images built from an image list file.

    This dataset is similar to the :class:`ImageList` dataset, except that this
    one uses an external text file that enumerates paths of images. The text
    file represents each example as one line, which may contains the label of
    the image. Paths and labels must be separated by whitespaces other than
    line break.

    Args:
        path (str): Path to the image-list file.
        withlabel (bool): If True, then it reads the image-list file as a list
            of pairs of image paths and labels. Otherwise, it assumes the
            image-list file is just a list of image paths.
        root (str): Root directory to retrieve images.
        dtype: Data type of resulting image arrays.

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
