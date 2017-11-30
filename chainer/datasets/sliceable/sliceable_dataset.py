import chainer


def _as_tuple(t):
    if isinstance(t, tuple):
        return t
    else:
        return t,


class SliceableDataset(chainer.dataset.DatasetMixin):

    def __len__(self):
        raise NotImplementedError

    @property
    def keys(self):
        raise NotImplementedError

    def get_example_by_keys(self, i, keys):
        raise NotImplementedError

    def get_example(self, i):
        if isinstance(self.keys, tuple):
            return self.get_example_by_keys(i, self.keys)
        else:
            return self.get_example_by_keys(i, (self.keys,))[0]

    @property
    def slice(self):
        return SliceHelper(self)


class SliceHelper(object):

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

    def get_example_by_keys(self, i, keys):
        if isinstance(self._index, slice):
            start, _, step = self._index.indices(len(self._dataset))
            return self._dataset.get_example_by_keys(start + i * step, keys)
        else:
            return self._dataset.get_example_by_keys(self._index[i], keys)
