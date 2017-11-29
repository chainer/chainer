import chainer


def _as_tuple(t):
    if isinstance(t, tuple):
        return t
    else:
        return t,


class BaseDataset(chainer.dataset.DatasetMixin):

    def __len__(self):
        raise NotImplementedError

    def get_example_by_keys(self, i, keys):
        raise NotImplementedError

    def get_example(self, i):
        return self.get_example_by_keys(i, self.keys)

    def pick(self, keys):
        for key in _as_tuple(keys):
            if key not in _as_tuple(self.keys):
                raise KeyError('{} does not exists'.format(key))
        return PickedDataset(self, keys)

    def sub(self, start, stop=None, step=None):
        return SubDataset(self, slice(start, stop, step))


class PickedDataset(BaseDataset):

    def __init__(self, base, keys):
        self._base = base
        self.keys = keys

    def __len__(self):
        return len(self._base)

    def get_example_by_keys(self, i, keys):
        return self._base.get_example_by_keys(i, keys)


class SubDataset(BaseDataset):

    def __init__(self, base, index):
        self._base = base
        self._index = index
        self.keys = base.keys

    def __len__(self):
        start, end, step = self._index.indices(len(self._base))
        return len(range(start, end, step))

    def get_example_by_keys(self, i, keys):
        start, _, step = self._index.indices(len(self._base))
        return self._base.get_example_by_keys(start + i * step, keys)


class PickableDataset(BaseDataset):

    def __init__(self):
        self._getters = dict()

    def __len__(self):
        raise NotImplementedError

    def add_getter(self, keys, getter):
        keys = _as_tuple(keys)
        for i, key in enumerate(keys):
            self._getters[key] = (getter, i)

    def get_example_by_keys(self, i, keys):
        if isinstance(keys, tuple):
            is_tuple = True
        else:
            keys = keys,
            is_tuple = False

        example = list()
        cache = dict()
        for key in keys:
            getter, index = self._getters[key]
            if getter not in cache:
                cache[getter] = _as_tuple(getter(i))
            example.append(cache[getter][index])

        if is_tuple:
            return tuple(example)
        else:
            return example[0]
