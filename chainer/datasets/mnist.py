import os

import numpy

import chainer
from chainer.dataset import download
from chainer.datasets._mnist_helper import make_npz
from chainer.datasets._mnist_helper import preprocess_mnist


def get_mnist(withlabel=True, ndim=1, scale=1., dtype=None,
              label_dtype=numpy.int32, rgb_format=False):
    """Gets the MNIST dataset.

    `MNIST <http://yann.lecun.com/exdb/mnist/>`_ is a set of hand-written
    digits represented by grey-scale 28x28 images. In the original images, each
    pixel is represented by one-byte unsigned integer. This function
    scales the pixels to floating point values in the interval ``[0, scale]``.

    This function returns the training set and the test set of the official
    MNIST dataset. If ``withlabel`` is ``True``, each dataset consists of
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
    train_raw = _retrieve_mnist_training()
    train = preprocess_mnist(train_raw, withlabel, ndim, scale, dtype,
                             label_dtype, rgb_format)
    test_raw = _retrieve_mnist_test()
    test = preprocess_mnist(test_raw, withlabel, ndim, scale, dtype,
                            label_dtype, rgb_format)
    return train, test


def _retrieve_mnist_training():
    urls = ['http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz']
    return _retrieve_mnist('train.npz', urls)


def _retrieve_mnist_test():
    urls = ['http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz']
    return _retrieve_mnist('test.npz', urls)


def _retrieve_mnist(name, urls):
    root = download.get_dataset_directory('pfnet/chainer/mnist')
    path = os.path.join(root, name)
    return download.cache_or_load_file(
        path, lambda path: make_npz(path, urls), numpy.load)
