from chainer.datasets.sliceable import SliceableDataset


class GetterDataset(SliceableDataset):
    """A sliceable dataset class that defined by getters.

    This ia a dataset class with getters.
    """

    def __init__(self):
        self._keys = []
        self._getters = {}

    def __len__(self):
        raise NotImplementedError

    @property
    def keys(self):
        return tuple(self._keys)

    @keys.setter
    def keys(self, keys):
        for key in keys:
            if key not in self._keys:
                raise KeyError('{} does not exists'.format(key))
        self._keys = keys

    def add_getter(self, keys, getter):
        """Register a getter function

        Args:
            keys (string or tuple of strings): Name(s) of data that
                the getter function returns.
            getter (callable): A getter function that takes an index and
                returns data of the corresponding example.
        """

        if isinstance(keys, tuple):
            self._keys += keys
            for i, key in enumerate(keys):
                self._getters[key] = (getter, i)
        else:
            self._keys.append(keys)
            self._getters[keys] = (getter, None)

    def get_example_by_keys(self, index, keys):
        example = []
        cache = {}
        for key in keys:
            getter, i = self._getters[key]
            if getter not in cache:
                cache[getter] = getter(index)
            if i is None:
                example.append(cache[getter])
            else:
                example.append(cache[getter][i])
        return tuple(example)
