import os
import shutil
import tempfile

from chainer import serializers


def save_and_load(src, dst, filename, saver, loader):
    """Saves ``src`` and loads it to ``dst`` using a de/serializer.

    This function simply runs a serialization and deserialization to check if
    the serialization code is correctly implemented. The save and load are
    done within a temporary directory.

    Args:
        src: An object to save from.
        dst: An object to load into.
        filename (str): File name used during the save/load.
        saver (callable): Function that saves the source object.
        loader (callable): Function that loads the file into the destination
            object.

    """
    tempdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tempdir, filename)
        saver(path, src)
        loader(path, dst)
    finally:
        shutil.rmtree(tempdir)


def save_and_load_npz(src, dst):
    """Saves ``src`` to an NPZ file and loads it to ``dst``.

    This is a short cut of :func:`save_and_load` using NPZ de/serializers.

    Args:
        src: An object to save.
        dst: An object to load to.

    """
    save_and_load(src, dst, 'tmp.npz',
                  serializers.save_npz, serializers.load_npz)


def save_and_load_hdf5(src, dst):
    """Saves ``src`` to an HDF5 file and loads it to ``dst``.

    This is a short cut of :func:`save_and_load` using HDF5 de/serializers.

    Args:
        src: An object to save.
        dst: An object to load to.

    """
    save_and_load(src, dst, 'tmp.h5',
                  serializers.save_hdf5, serializers.load_hdf5)
