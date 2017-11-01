import gzip
import struct

import numpy
import six

from chainer.dataset import download
from chainer.datasets import tuple_dataset


def make_npz(path, urls):
    x_url, y_url = urls
    x_path = download.cached_download(x_url)
    y_path = download.cached_download(y_url)

    with gzip.open(x_path, 'rb') as fx, gzip.open(y_path, 'rb') as fy:
        fx.read(4)
        fy.read(4)
        N, = struct.unpack('>i', fx.read(4))
        if N != struct.unpack('>i', fy.read(4))[0]:
            raise RuntimeError('wrong pair of MNIST images and labels')
        fx.read(8)

        x = numpy.empty((N, 784), dtype=numpy.uint8)
        y = numpy.empty(N, dtype=numpy.uint8)

        for i in six.moves.range(N):
            y[i] = ord(fy.read(1))
            for j in six.moves.range(784):
                x[i, j] = ord(fx.read(1))

    numpy.savez_compressed(path, x=x, y=y)
    return {'x': x, 'y': y}


def preprocess_mnist(raw, withlabel, ndim, scale, image_dtype, label_dtype,
                     rgb_format):
    images = raw['x']
    if ndim == 2:
        images = images.reshape(-1, 28, 28)
    elif ndim == 3:
        images = images.reshape(-1, 1, 28, 28)
        if rgb_format:
            images = numpy.broadcast_to(images,
                                        (len(images), 3) + images.shape[2:])
    elif ndim != 1:
        raise ValueError('invalid ndim for MNIST dataset')
    images = images.astype(image_dtype)
    images *= scale / 255.

    if withlabel:
        labels = raw['y'].astype(label_dtype)
        return tuple_dataset.TupleDataset(images, labels)
    else:
        return images
