from chainer.datasets.sliceable import SliceableDataset


class GetterDataset(SliceableDataset):

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
        for key in keys:
            if key not in self._keys:
                raise KeyError('{} does not exists'.format(key))
        self._keys = keys

    def add_getter(self, keys, getter):
        if isinstance(keys, tuple):
            self._keys += keys
            for j, key in enumerate(keys):
                self._getters[key] = (getter, j)
        else:
            self._keys += (keys,)
            self._getters[keys] = (getter, None)

    def get_example_by_keys(self, i, keys):
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
        return tuple(example)
