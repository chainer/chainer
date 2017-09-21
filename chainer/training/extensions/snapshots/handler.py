import h5py
import numpy

from chainer.serializers import npz


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


class SerializerHandler(object):
    """Base class of handler of serializers.

    This handler is used in snapshot extension of
    :class:`~chainer.training.Trainer`. To divide the timinig of serialization
    and actual saving, this class provides :func:`serialize()` and
    :func:`save` methods.

    Args:
        savefun: Callable object. It is invoked in :func:`save()` and does the
            actual saving to the file.
        kwds: Keyword arguments that are passed to ``savefun``.
    """

    def __init__(self, savefun=save_npz, **kwds):
        self.savefun = savefun
        self.kwds = kwds

    def serialize(self, target):
        """Serialize the given target.

        This method creates a standard serializer in Chainer and serialize
        a target using this serializer.

        Args:
            target: Object to be serialize. Usually it is a trainer object.
        """
        self.serializer = npz.DictionarySerializer()
        self.serializer.save(target)
        self.target = self.serializer.target

    def save(self, filename):
        """Save the serialized target with a given file name.

        This method actually saves the registered serialized target with a
        given file name.
        """
        self.savefun(filename, self.target, **self.kwds)
