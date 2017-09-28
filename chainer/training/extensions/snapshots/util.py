import h5py
import numpy

from chainer.serializers import npz


def serialize(target):
    serializer = npz.DictionarySerializer()
    serializer.save(target)
    return serializer.target


def save_npz(filename, target, compression=True):
    with open(filename, 'wb') as f:
        if compression:
            numpy.savez_compressed(f, **target)
        else:
            numpy.savez(f, **target)


def save_hdf5(filename, target, compression=4):
    with h5py.File(filename, 'w') as f:
        for key, value in target.items():
            key = '/' + key.lstrip('/')
            arr = numpy.asarray(value)
            compression = None if arr.size <= 1 else compression
            try:
                f.create_dataset(key, data=arr, compression=compression)
            except TypeError:
                # '/extensions/LogReport/_log'
                # '/extensions/PlotReport/_plot_loss.png'
                # '/extensions/PlotReport/_plot_accuracy.png'
                # Keys above are not be able to be saved in HDF5 format.
                # numpy.dtype('O') is the datatype of these key's values and
                # it is not supported in h5py.
                pass


def load_npz(filename):
    target = {}
    with numpy.load(filename) as f:
        for key in f.files:
            target[key] = f[key]
    return target


def load_hdf5(filename):
    target = {}
    with h5py.File(filename, 'r') as f:
        for key, value in f.items():
            target[key] = value
    return target
