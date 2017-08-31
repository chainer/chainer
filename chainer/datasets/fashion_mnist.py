import gzip
import os
import struct

import numpy
import six

from chainer.dataset import download
from chainer.datasets import tuple_dataset
from chainer.datasets.mnist import _preprocess_mnist, _make_npz

def get_fashion_mnist(withlabel=True, ndim=1, scale=1., dtype=numpy.float32,
              label_dtype=numpy.int32):
    """Gets the Fashion-MNIST dataset.

    `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist/>`_ is a set 
    of fashion articles represented by grey-scale 28x28 images. In the original 
    images, each pixel is represented by one-byte unsigned integer. This function
    scales the pixels to floating point values in the interval ``[0, scale]``.

    This function returns the training set and the test set of the official
    Fashion-MNIST dataset. If ``withlabel`` is ``True``, each dataset consists of
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
        dtype: Data type of resulting image arrays.
        label_dtype: Data type of the labels.

    Returns:
        A tuple of two datasets. If ``withlabel`` is ``True``, both datasets
        are :class:`~chainer.datasets.TupleDataset` instances. Otherwise, both
        datasets are arrays of images.

    """
    train_raw = _retrieve_fashion_mnist_training()
    train = _preprocess_mnist(train_raw, withlabel, ndim, scale, dtype,
                              label_dtype)
    test_raw = _retrieve_fashion_mnist_test()
    test = _preprocess_mnist(test_raw, withlabel, ndim, scale, dtype,
                             label_dtype)
    return train, test

def _retrieve_fashion_mnist_training():
    urls = ['http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz']
    return _retrieve_fashion_mnist('train.npz', urls)


def _retrieve_fashion_mnist_test():
    urls = ['http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
            'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz']
    return _retrieve_fashion_mnist('test.npz', urls)

def _retrieve_fashion_mnist(name, urls):
    root = download.get_dataset_directory('pfnet/chainer/fashion-mnist')
    path = os.path.join(root, name)
    return download.cache_or_load_file(
        path, lambda path: _make_npz(path, urls), numpy.load)