import chainer


def _as_tuple(t):
    if isinstance(t, tuple):
        return t
    else:
        return t,


class SliceableDataset(chainer.dataset.DatasetMixin):
    """An abstract dataset class that supports slicing.

    This ia a dataset class that supports slicing.
    A dataset class inheriting this class should implement
    three methods: :meth:`__len__`, :meth:`keys`, and
    :meth:`get_example_by_keys`.
    """

    def __len__(self):
        raise NotImplementedError

    @property
    def keys(self):
        """Return all keys

        Returns:
            string or tuple of strings
        """
        raise NotImplementedError

    def get_example_by_keys(self, index, keys):
        """Return data of an example by keys

        Args:
            index (int): An index of an example.
            keys (tuple of strings): A tuple of requested keys.

        Returns:
            tuple of data
        """
        raise NotImplementedError

    def get_example(self, index):
        if isinstance(self.keys, tuple):
            return self.get_example_by_keys(index, self.keys)
        else:
            return self.get_example_by_keys(index, (self.keys,))[0]

    @property
    def slice(self):
        return SliceHelper(self)


class SliceHelper(object):
    """A helper class for :class:`SliceableDataset`."""

    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, args):
        if isinstance(args, tuple):
            index, keys = args
        else:
            index = args
            keys = self._dataset.keys
        for key in _as_tuple(keys):
            if key not in _as_tuple(self._dataset.keys):
                raise KeyError('{} does not exists'.format(key))
        return SlicedDataset(self._dataset, index, keys)


class SlicedDataset(SliceableDataset):
    """A sliced view for :class:`SliceableDataset`."""

    def __init__(self, dataset, index, keys):
        self._dataset = dataset
        self._index = index
        self._keys = keys

    def __len__(self):
        if isinstance(self._index, slice):
            start, end, step = self._index.indices(len(self._dataset))
            return len(range(start, end, step))
        else:
            return len(self._index)

    @property
    def keys(self):
        return self._keys

    def get_example_by_keys(self, index, keys):
        if isinstance(self._index, slice):
            start, _, step = self._index.indices(len(self._dataset))
            return self._dataset.get_example_by_keys(
                start + index * step, keys)
        else:
            return self._dataset.get_example_by_keys(self._index[index], keys)
