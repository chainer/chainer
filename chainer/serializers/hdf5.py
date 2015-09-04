import h5py
import numpy

import chainer
from chainer import cuda


class HDF5Serializer(chainer.Serializer):

    writer = True

    def __init__(self, group, compress=4):
        self.group = group
        self.compress = compress

    def __getitem__(self, key):
        return HDF5Serializer(self.group.require_group(key), self.compress)

    def __call__(self, key, value):
        if isinstance(value, cuda.ndarray):
            arr = value.get()
        else:
            arr = numpy.asarray(value)

        if self.compress:
            self.group.create_dataset(key, data=arr, compress=self.compress)
        else:
            self.group.create_dataset(key, data=arr)

        return value


def save_h5py(filename, obj, compress=4):
    f = h5py.File(filename, 'w')
    s = HDF5Serializer(f, compress)
    obj.serialize(s)


class HDF5Deserializer(chainer.Serializer):

    writer = False

    def __init__(self, group):
        self.group = group

    def __getitem__(self, key):
        return HDF5Deserializer(self.group[key])

    def __call__(self, key, value):
        dataset = self.group[key]
        if isinstance(value, numpy.ndarray):
            dataset.read_direct(value)
        elif isinstance(value, cuda.ndarray):
            value.set(numpy.asarray(dataset))
        else:
            value = type(value)(dataset)
        return value


def load_h5py(filename, obj):
    f = h5py(filename, 'r')
    s = HDF5Deserializer(f)
    obj.serialize(s)
