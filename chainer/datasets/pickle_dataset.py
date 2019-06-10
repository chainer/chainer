import threading

import six
import six.moves.cPickle as pickle

from chainer.dataset import dataset_mixin


class PickleDatasetWriter(object):

    """Writer class that makes PickleDataset.

    To make :class:`PickleDataset`, a user needs to prepare data using
    :class:`PickleDatasetWriter`.

    Args:
        writer: File like object that supports ``write`` and ``tell`` methods.
        protocol (int): Valid protocol for :mod:`pickle`.

    .. seealso: chainer.datasets.PickleDataset

    """

    def __init__(self, writer, protocol=pickle.HIGHEST_PROTOCOL):
        self._positions = []
        self._writer = writer
        self._protocol = protocol

    def close(self):
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, x):
        position = self._writer.tell()
        pickle.dump(x, self._writer, protocol=self._protocol)
        self._positions.append(position)

    def flush(self):
        if hasattr(self._writer, 'flush'):
            self._writer.flush()


class PickleDataset(dataset_mixin.DatasetMixin):

    """Dataset stored in a storage using pickle.

    :mod:`pickle` is the default serialization library of Python.
    This dataset stores any objects in a storage using :mod:`pickle`.
    Even when a user wants to use a large dataset, this dataset can stores all
    data in a large storage like HDD and each data can be randomly accessible.

    .. testsetup::

        import tempfile
        fs, path_to_data = tempfile.mkstemp()

    >>> with chainer.datasets.open_pickle_dataset_writer(path_to_data) as w:
    ...     w.write((1, 2.0, 'hello'))
    ...     w.write((2, 3.0, 'good-bye'))
    ...
    >>> with chainer.datasets.open_pickle_dataset(path_to_data) as dataset:
    ...     print(dataset[1])
    ...
    (2, 3.0, 'good-bye')

    .. testcleanup::

        import os
        os.close(fs)

    Args:
        reader: File like object. `reader` must support random access.

    """

    def __init__(self, reader):
        # Only py3 supports `seekable` method
        if six.PY3 and not reader.seekable():
            raise ValueError('reader must support random access')
        self._reader = reader
        self._positions = []
        reader.seek(0)
        while True:
            position = reader.tell()
            try:
                pickle.load(reader)
            except EOFError:
                break
            self._positions.append(position)

        self._lock = threading.RLock()

    def close(self):
        """Closes a file reader.

        After a user calls this method, the dataset will no longer be
        accessible..
        """
        with self._lock:
            self._reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __len__(self):
        return len(self._positions)

    def get_example(self, index):
        with self._lock:
            self._reader.seek(self._positions[index])
            return pickle.load(self._reader)


def open_pickle_dataset(path):
    """Opens a dataset stored in a given path.

    This is a helper function to open :class:`PickleDataset`. It opens a given
    file in binary mode, and creates a :class:`PickleDataset` instance.

    This method does not close the opened file. A user needs to call
    :func:`PickleDataset.close` or use `with`:

    .. code-block:: python

        with chainer.datasets.open_pickle_dataset('path') as dataset:
            pass  # use dataset

    Args:
        path (str): Path to a dataset.

    Returns:
        chainer.datasets.PickleDataset: Opened dataset.

    .. seealso: chainer.datasets.PickleDataset

    """
    reader = open(path, 'rb')
    return PickleDataset(reader)


def open_pickle_dataset_writer(path, protocol=pickle.HIGHEST_PROTOCOL):
    """Opens a writer to make a PickleDataset.

    This is a helper function to open :class:`PickleDatasetWriter`. It opens a
    given file in binary mode and creates a :class:`PickleDatasetWriter`
    instance.

    This method does not close the opened file. A user needs to call
    :func:`PickleDatasetWriter.close` or use `with`:

    .. code-block:: python

        with chainer.datasets.open_pickle_dataset_writer('path') as writer:
            pass  # use writer

    Args:
        path (str): Path to a dataset.
        protocol (int): Valid protocol for :mod:`pickle`.

    Returns:
        chainer.datasets.PickleDatasetWriter: Opened writer.

    .. seealso: chainer.datasets.PickleDataset

    """
    writer = open(path, 'wb')
    return PickleDatasetWriter(writer, protocol=protocol)
