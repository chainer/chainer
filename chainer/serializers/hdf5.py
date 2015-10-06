import h5py
import numpy

import chainer
from chainer import cuda


class HDF5Serializer(chainer.Serializer):

    """Serializer in HDF5 format.

    This is the standard serializer of Chainer. The hierarchy is simply
    represented by HDF5 groups.

    Args:
        group (h5py.Group): The group that this serializer represents.
        compression (int): Gzip compression level.

    """
    writer = True

    def __init__(self, group, compression=4):
        self.group = group
        self.compression = compression

    def __getitem__(self, key):
        return HDF5Serializer(self.group.require_group(key), self.compression)

    def __call__(self, key, value):
        if isinstance(value, cuda.ndarray):
            arr = value.get()
        else:
            arr = numpy.asarray(value)

        if self.compression and isinstance(value, numpy.ndarray):
            self.group.create_dataset(key, data=arr,
                                      compression=self.compression)
        else:
            self.group.create_dataset(key, data=arr)

        return value


def save_hdf5(filename, obj, compression=4):
    """Saves a serializable object to specified file in HDF5 format.

    This is an easy short-cut API to save a file consisting of only one
    serializable object. In order to save multiple serializable objects into
    one HDF5 file, use the :class:`HDF5Serializer` class directly.

    Args:
        filename (str): Path to the HDF5 file to be saved.
        obj (serializable): Target object.
        compression (int): Gzip compression level.

    """
    f = h5py.File(filename, 'w')
    s = HDF5Serializer(f, compression)
    obj.serialize(s)


class HDF5Deserializer(chainer.Serializer):

    """Deserializer in HDF5 format.

    This deserializer is used to load objects saved by :class:`HDF5Serializer`.

    Args:
        group (h5py.Group): The group that this deserializer represents.

    """
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
            value = type(value)(numpy.asarray(dataset))
        return value


def load_hdf5(filename, obj):
    """Loads a serializable object from specified file in HDF5 format.

    This is an easy short-cut API to load from a file consisting of only one
    serializable object. In order to load multiple serializable objects from
    one HDF5 file, use the :class:`HDF5Deserializer` class directly.

    Args:
        filename (str): Path to the HDF5 file to be loaded.
        obj (serializable): Target object.

    """
    f = h5py.File(filename, 'r')
    s = HDF5Deserializer(f)
    obj.serialize(s)
