from __future__ import print_function
import os
import tarfile

import numpy
import six.moves.cPickle as pickle
from six.moves.urllib import request

from chainer import dataset


class CifarBase(dataset.Dataset):

    def __init__(self, dims=3, dtype=numpy.float32, scale=1):
        self._data, self._labels = _load(
            self._dataname, self._type, dims, dtype, scale)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, i):
        return self._data[i], self._labels[i]

    def compute_mean(self):
        return self._data.mean(axis=0)


class Cifar10Training(CifarBase):

    """CIFAR-10 training dataset.

    TODO(beam2d): document it.

    """
    name = 'CIFAR10 train'
    _dataname = 'CIFAR10'
    _type = 'train'


class Cifar10Test(CifarBase):

    """CIFAR-10 test dataset.

    TODO(beam2d): document it.

    """
    name = 'CIFAR10 test'
    _dataname = 'CIFAR10'
    _type = 'test'


class Cifar100Training(CifarBase):

    """CIFAR-100 training dataset.

    TODO(beam2d): document it.

    """
    name = 'CIFAR100 training'
    _dataname = 'CIFAR100'
    _type = 'train'


class Cifar100Test(CifarBase):

    """CIFAR-100 test dataset.

    TODO(beam2d): document it.

    """
    name = 'CIFAR100 test'
    _dataname = 'CIFAR100'
    _type = 'test'


_urls = {
    'CIFAR10': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
    'CIFAR100': 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
}


def _get_path(root, dataname, typ):
    return os.path.join(root, '{}_{}.npz'.format(dataname, typ))


def _load(dataname, typ, dims, dtype, scale):
    root = dataset.get_dataset_path('cifar')
    path = _get_path(root, dataname, typ)

    if dims == 1:
        shape = -1, 3072
    elif dims == 3:
        shape = -1, 3, 32, 32
    else:
        raise ValueError('CIFAR dims must be 1 or 3')

    try:
        npz = numpy.load(path)
        data = npz['x']
        labels = npz['y']
    except IOError:
        print('Downloading {} dataset...'.format(dataname))
        try:
            os.makedirs(root)
        except OSError:
            pass
        data, labels = _download(root, dataname, typ)

    data = data.reshape(shape).astype(dtype)
    data *= scale / 255.
    return data, labels.astype(numpy.int32)


def _download(root, dataname, typ):
    rawpath = os.path.join(root, dataname + '.raw')
    request.urlretrieve(_urls[dataname], rawpath)

    train_x = numpy.empty((5, 10000, 3072), dtype=numpy.uint8)
    train_y = numpy.empty((5, 10000), dtype=numpy.uint8)
    test_y = numpy.empty(10000, dtype=numpy.uint8)

    with tarfile.open(rawpath, 'r:gz') as archive:
        # training set
        for i in range(5):
            filename = _get_batch_name(dataname, 'data_batch_{}'.format(i + 1))
            d = pickle.load(archive.extractfile(filename))
            train_x[i] = d['data']
            train_y[i] = d['labels']

        # test set
        filename = _get_batch_name(dataname, 'test_batch')
        d = pickle.load(archive.extractfile(filename))
        test_x = d['data']  # just use the loaded array
        test_y[...] = d['labels']

    train_x = train_x.reshape(50000, 3072)
    train_y = train_y.reshape(50000)

    os.remove(rawpath)
    numpy.savez_compressed(_get_path(root, dataname, 'train'),
                           x=train_x, y=train_y)
    numpy.savez_compressed(_get_path(root, dataname, 'test'),
                           x=test_x, y=test_y)

    if typ == 'train':
        return train_x.reshape(50000, 3072), train_y.reshape(50000)
    else:
        return test_x, test_y


def _get_batch_name(dataname, name):
    if dataname == 'CIFAR10':
        return 'cifar-10-batches-py/' + name
    else:
        return 'cifar-100-batches-py/' + name
