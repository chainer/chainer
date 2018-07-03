import os

from PIL import Image
import random
import six

import numpy


def _read_image_as_array(path, dtype):
    f = Image.open(path)
    try:
        image = numpy.asarray(f, dtype=dtype)
    finally:
        # Only pillow >= 3.0 has 'close' method
        if hasattr(f, 'close'):
            f.close()
    return image


def _postprocess_image(image):
    if image.ndim == 2:
        # image is greyscale
        image = image[..., None]
    return image.transpose(2, 0, 1)


class LabeledImageDataset:

    """Dataset of image and label pairs built from a list of paths and labels.

    This dataset reads an external image file like :class:`ImageDataset`. The
    difference from :class:`ImageDataset` is that this dataset also returns a
    label integer. The paths and labels are given as either a list of pairs or
    a text file contains paths/labels pairs in distinct lines. In the latter
    case, each path and corresponding label are separated by white spaces. This
    format is same as one used in Caffe.

    .. note::
       **This dataset requires the Pillow package being installed.** In order
       to use this dataset, install Pillow (e.g. by using the command ``pip
       install Pillow``). Be careful to prepare appropriate libraries for image
       formats you want to use (e.g. libpng for PNG images, and libjpeg for JPG
       images).

    .. warning::
       **You are responsible for preprocessing the images before feeding them
       to a model.** For example, if your dataset contains both RGB and
       grayscale images, make sure that you convert them to the same format.
       Otherwise you will get errors because the input dimensions are different
       for RGB and grayscale images.

    Args:
        pairs (str or list of tuples): If it is a string, it is a path to a
            text file that contains paths to images in distinct lines. If it is
            a list of pairs, the ``i``-th element represents a pair of the path
            to the ``i``-th image and the corresponding label. In both cases,
            each path is a relative one from the root path given by another
            argument.
        root (str): Root directory to retrieve images from.
        label_dtype: Data type of the labels.

    """

    def __init__(self, pairs, root='.', label_dtype=numpy.int32):
        if isinstance(pairs, six.string_types):
            pairs_path = pairs
            with open(pairs_path) as pairs_file:
                pairs = []
                for i, line in enumerate(pairs_file):
                    pair = line.strip().split()
                    if len(pair) != 2:
                        raise ValueError(
                            'invalid format at line {} in file {}'.format(
                                i, pairs_path))
                    pairs.append((pair[0], int(pair[1])))
        self._pairs = pairs
        self._root = root
        self._label_dtype = label_dtype
        self._dtype = numpy.float32

    def __len__(self):
        return len(self._pairs)

    def get_example(self, i):
        path, int_label = self._pairs[i]
        full_path = os.path.join(self._root, path)
        image = _read_image_as_array(full_path, self._dtype)

        label = numpy.array(int_label, dtype=self._label_dtype)
        return _postprocess_image(image), label


class PreprocessedDataset:

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = LabeledImageDataset(path, root)
        self.mean = mean.astype(numpy.float32)
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base.get_example(i)
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label


def get_dataset(path, root, mean, crop_size, random=True):
    pd = PreprocessedDataset(path, root, mean, crop_size, random)
    X = []
    Y = []
    for i in six.moves.range(len(pd)):
        image, label = pd.get_example(i)
        X.append(image)
        Y.append(label)
    return X, Y
