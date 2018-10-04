import os

import numpy

import chainer
from chainer.dataset import download
from chainer.datasets._mnist_helper import make_npz
from chainer.datasets._mnist_helper import preprocess_mnist


_fashion_mnist_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def get_fashion_mnist_labels():
    """Provide a list of the string value names of the labels.

    Returns:
        List of string values of the image labels.

    """
    return list(_fashion_mnist_labels)


def get_fashion_mnist(withlabel=True, ndim=1, scale=1., dtype=None,
                      label_dtype=numpy.int32, rgb_format=False):
    """Gets the Fashion-MNIST dataset.

    `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist/>`_ is a
    set of fashion articles represented by grey-scale 28x28 images. In the
    original images, each pixel is represented by one-byte unsigned integer.
    This function scales the pixels to floating point values in the interval
    ``[0, scale]``.

    This function returns the training set and the test set of the official
    Fashion-MNIST dataset. If ``withlabel`` is ``True``, each dataset consists
    of tuples of images and labels, otherwise it only consists of images.

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
    train_raw = _retrieve_fashion_mnist_training()
    dtype = chainer.get_dtype(dtype)

    train = preprocess_mnist(train_raw, withlabel, ndim, scale, dtype,
                             label_dtype, rgb_format)
    test_raw = _retrieve_fashion_mnist_test()
    test = preprocess_mnist(test_raw, withlabel, ndim, scale, dtype,
                            label_dtype, rgb_format)
    return train, test


def _retrieve_fashion_mnist_training():
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    urls = [base_url + 'train-images-idx3-ubyte.gz',
            base_url + 'train-labels-idx1-ubyte.gz']
    return _retrieve_fashion_mnist('train.npz', urls)


def _retrieve_fashion_mnist_test():
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    urls = [base_url + 't10k-images-idx3-ubyte.gz',
            base_url + 't10k-labels-idx1-ubyte.gz']
    return _retrieve_fashion_mnist('test.npz', urls)


def _retrieve_fashion_mnist(name, urls):
    root = download.get_dataset_directory('pfnet/chainer/fashion-mnist')
    path = os.path.join(root, name)
    return download.cache_or_load_file(
        path, lambda path: make_npz(path, urls), numpy.load)
