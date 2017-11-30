import chainer


def _as_tuple(t):
    if isinstance(t, tuple):
        return t
    else:
        return t,


class BaseDataset(chainer.dataset.DatasetMixin):

    def __len__(self):
        raise NotImplementedError

    @property
    def keys(self):
        raise NotImplementedError

    def get_example_by_keys(self, i, keys):
        raise NotImplementedError

    def get_example(self, i):
        return self.get_example_by_keys(i, self.keys)

    @property
    def slice(self):
        return SlicedDatasetHelper(self)

    def concatenate(self, *datasets):
        for dataset in datasets:
            if not dataset.keys == self.keys:
                raise ValueError('mismatched keys')
        return ConcatenatedDataset((self,) + datasets)

    def transform(self, func, keys):
        return TransformedDataset(self, func, keys)


class SlicedDatasetHelper(object):
    def __init__(self, base):
        self._base = base

    def __getitem__(self, args):
        if isinstance(args, tuple):
            index, keys = args
        else:
            index = args
            keys = self._base.keys
        for key in _as_tuple(keys):
            if key not in _as_tuple(self._base.keys):
                raise KeyError('{} does not exists'.format(key))
        return SlicedDataset(self._base, index, keys)


class SlicedDataset(BaseDataset):

    def __init__(self, base, index, keys):
        self._base = base
        self._index = index
        self._keys = keys

    def __len__(self):
        if isinstance(self._index, slice):
            start, end, step = self._index.indices(len(self._base))
            return len(range(start, end, step))
        else:
            return len(self._index)

    @property
    def keys(self):
        return self._keys

    def get_example_by_keys(self, i, keys):
        if isinstance(self._index, slice):
            start, _, step = self._index.indices(len(self._base))
            return self._base.get_example_by_keys(start + i * step, keys)
        else:
            return self._base.get_example_by_keys(self._index[i], keys)


class ConcatenatedDataset(BaseDataset):

    def __init__(self, datasets):
        self._datasets = datasets
        self._keys = datasets[0].keys

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    @property
    def keys(self):
        return self._keys

    def get_example_by_keys(self, i, keys):
        if i < 0:
            raise IndexError
        for dataset in self._datasets:
            if i < len(dataset):
                return dataset.get_example_by_keys(i, keys)
            i -= len(dataset)
        raise IndexError


class PickableDataset(BaseDataset):

    def __init__(self):
        self._keys = ()
        self._getters = {}

    def __len__(self):
        raise NotImplementedError

    @property
    def keys(self):
        return self._keys

    @keys.setter
    def keys(self, keys):
        for key in _as_tuple(keys):
            if key not in _as_tuple(self._keys):
                raise KeyError('{} does not exists'.format(key))
        self._keys = keys

    def add_getter(self, getter, keys):
        if isinstance(keys, tuple):
            self._keys += keys
            for j, key in enumerate(keys):
                self._getters[key] = (getter, j)
        else:
            self._keys += (keys,)
            self._getters[keys] = (getter, None)

    def get_example_by_keys(self, i, keys):
        if isinstance(keys, tuple):
            is_tuple = True
        else:
            keys = keys,
            is_tuple = False

        example = []
        cache = {}
        for key in keys:
            getter, j = self._getters[key]
            if getter not in cache:
                cache[getter] = getter(i)
            if j is None:
                example.append(cache[getter])
            else:
                example.append(cache[getter][j])

        if is_tuple:
            return tuple(example)
        else:
            return example[0]


class TransformedDataset(PickableDataset):

    def __init__(self, base, func, keys):
        super(TransformedDataset, self).__init__()
        self._base = base
        self.add_getter(lambda i: func(base.get_example(i)), keys)

    def __len__(self):
        return len(self._base)
