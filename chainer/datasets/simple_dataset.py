from chainer import dataset


class SimpleDataset(dataset.Dataset):

    """Simple dataset based on a design matrix.

    TODO(beam2d): document it.

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
