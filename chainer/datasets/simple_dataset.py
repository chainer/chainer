from chainer import dataset


class SimpleDataset(dataset.Dataset):

    """Simple dataset based on a design matrix.

    This dataset provides a Dataset interface on general arrays. Each array
    represents a set of examples stacked along the first axis. If a tuple of
    arrays are given, then the ``__getitem__`` operator returns a tuple of
    same numbers of values.

    Args:
        name (str): Name of the dataset. This value is set to the :attr:`name`
            attribute.
        arrays (array or tuple of arrays): Base arrays whose columns represent
            the data points in the dataset.

    Attribute:
        name: Name of the dataset.

    """
    def __init__(self, name, arrays):
        self.name = name
        self._arrays = arrays
        self._ret_tuple = isinstance(arrays, tuple)
        if self._ret_tuple:
            self._len = len(arrays[0])
            if any(len(a) != self._len for a in arrays):
                raise ValueError('arrays have different numbers of examples')
        else:
            self._len = len(arrays)

    def __len__(self):
        return self._len

    def __getitem__(self, i):
        if self._ret_tuple:
            return tuple([a[i] for a in self._arrays])
        else:
            return self._arrays[i]
