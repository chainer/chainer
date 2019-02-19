import os

import numpy

import chainer
from chainer.dataset import download
from chainer.datasets._mnist_helper import make_npz
from chainer.datasets._mnist_helper import preprocess_mnist


_kuzushiji_mnist_labels = [('o', u'\u304A'), ('ki', u'\u304D'),
                           ('su', u'\u3059'), ('tsu', u'\u3064'),
                           ('na', u'\u306A'), ('ha', u'\u306F'),
                           ('ma', u'\u307E'), ('ya', u'\u3084'),
                           ('re', u'\u308C'), ('wo', u'\u3092')]


def get_kuzushiji_mnist_labels():
    """Provides a list of labels for the Kuzushiji-MNIST dataset.

    Returns:
        List of labels in the form of tuples. Each tuple contains the
        character name in romaji as a string value and the unicode codepoint
        for the character.

    """
    return _kuzushiji_mnist_labels


def get_kuzushiji_mnist(withlabel=True, ndim=1, scale=1., dtype=None,
                        label_dtype=numpy.int32, rgb_format=False):
    """Gets the Kuzushiji-MNIST dataset.

    `Kuzushiji-MNIST (KMNIST) <http://codh.rois.ac.jp/kmnist/>`_ is a set of
    hand-written Japanese characters represented by grey-scale 28x28 images.
    In the original images, each pixel is represented by one-byte unsigned
    integer. This function scales the pixels to floating point values in the
    interval ``[0, scale]``.

    This function returns the training set and the test set of the official
    KMNIST dataset. If ``withlabel`` is ``True``, each dataset consists of
    tuples of images and labels, otherwise it only consists of images.

    Args:
        withlabel (bool): If ``True``, it returns datasets with labels. In this
            case, each example is a tuple of an image and a label. Otherwise,
            the datasets only contain images.
        ndim (int): Number of dimensions of each image. The shape of each image
            is determined depending on ``ndim`` as follows:

            - ``ndim == 1``: the shape is ``(784,)``
            - ``ndim == 2``: the shape is ``(28, 28)``
            - ``ndim == 3``: the shape is ``(1, 28, 28)``

        scale (float): Pixel value scale. If it is 1 (default), pixels are
            scaled to the interval ``[0, 1]``.
        dtype: Data type of resulting image arrays. ``chainer.config.dtype`` is
            used by default (see :ref:`configuration`).
        label_dtype: Data type of the labels.
        rgb_format (bool): if ``ndim == 3`` and ``rgb_format`` is ``True``, the
            image will be converted to rgb format by duplicating the channels
            so the image shape is (3, 28, 28). Default is ``False``.

    Returns:
        A tuple of two datasets. If ``withlabel`` is ``True``, both datasets
        are :class:`~chainer.datasets.TupleDataset` instances. Otherwise, both
        datasets are arrays of images.

    """
    dtype = chainer.get_dtype(dtype)
    train_raw = _retrieve_kuzushiji_mnist_training()
    train = preprocess_mnist(train_raw, withlabel, ndim, scale, dtype,
                             label_dtype, rgb_format)
    test_raw = _retrieve_kuzushiji_mnist_test()
    test = preprocess_mnist(test_raw, withlabel, ndim, scale, dtype,
                            label_dtype, rgb_format)
    return train, test


def _retrieve_kuzushiji_mnist_training():
    base_url = 'http://codh.rois.ac.jp/'
    urls = [base_url + 'kmnist/dataset/kmnist/train-images-idx3-ubyte.gz',
            base_url + 'kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz']
    return _retrieve_kuzushiji_mnist('train.npz', urls)


def _retrieve_kuzushiji_mnist_test():
    base_url = 'http://codh.rois.ac.jp/'
    urls = [base_url + 'kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz',
            base_url + 'kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz']
    return _retrieve_kuzushiji_mnist('test.npz', urls)


def _retrieve_kuzushiji_mnist(name, urls):
    root = download.get_dataset_directory('pfnet/chainer/kuzushiji_mnist')
    path = os.path.join(root, name)
    return download.cache_or_load_file(
        path, lambda path: make_npz(path, urls), numpy.load)
