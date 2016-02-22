import gzip
import os

import numpy
import six
from six.moves.urllib import request

from chainer import dataset


class MnistBase(dataset.Dataset):

    def __init__(self, dims=1, dtype=numpy.float32, scale=1):
        self._data, self._labels = _load(self._type, dims, dtype, scale)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, i):
        return (self._data[i], self._labels[i])


class MnistTraining(MnistBase):

    """MNIST training dataset.

    TODO(beam2d): document it.

    """
    name = 'MNIST_train'
    _type = 'train'


class MnistTest(MnistBase):

    """MNIST test dataset.

    TODO(beam2d): document it.

    """
    name = 'MNIST_test'
    _type = 'test'


_info = {
    'train': {
        'x_url': 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'y_url': 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'n': 60000,
    },
    'test': {
        'x_url': 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'y_url': 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
        'n': 10000,
    }
}


def _load(name, dims, dtype, scale):
    info = _info[name]
    n = info['n']

    if dims == 1:
        shape = n, 784
    elif dims == 2:
        shape = n, 28, 28
    elif dims == 3:
        shape = n, 1, 28, 28
    else:
        raise ValueError('MNIST dims must be 1, 2, or 3')

    root = dataset.get_dataset_path('mnist')
    path = os.path.join(root, name + '.npz')

    try:
        npz = numpy.load(path)
        data = npz['x']
        labels = npz['y']
    except IOError:
        print('Downloading MNIST {} dataset...'.format(name))
        try:
            os.makedirs(root)
        except OSError:
            pass
        data, labels = _download(path, info)

    data = data.reshape(shape).astype(dtype)
    data *= scale / 255.
    return data, labels.astype(numpy.int32)


def _download(path, info):
    x_rawpath = path + '-data.raw'
    y_rawpath = path + '-labels.raw'
    request.urlretrieve(info['x_url'], x_rawpath)
    request.urlretrieve(info['y_url'], y_rawpath)

    n = info['n']
    x = numpy.zeros((n, 784), dtype=numpy.uint8)
    y = numpy.empty(n, dtype=numpy.uint8)

    with gzip.open(x_rawpath, 'rb') as fx, gzip.open(y_rawpath, 'rb') as fy:
        fx.read(16)
        fy.read(8)
        for i in six.moves.range(n):
            y[i] = ord(fy.read(1))
            for j in six.moves.range(784):
                x[i, j] = ord(fx.read(1))

    os.remove(x_rawpath)
    os.remove(y_rawpath)
    numpy.savez_compressed(path, x=x, y=y)
    return x, y
