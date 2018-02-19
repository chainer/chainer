from chainer.datasets.sliceable import SliceableDataset


def _as_tuple(t):
    if isinstance(t, tuple):
        return t
    else:
        return t,


class GetterDataset(SliceableDataset):
    """A sliceable dataset class that defined by getters.

    This ia a dataset class with getters.
    """

    def __init__(self):
        self._keys = []
        self._getters = []

    def __len__(self):
        raise NotImplementedError

    @property
    def keys(self):
        return tuple(key for key, _, _ in self._keys)

    def add_getter(self, keys, getter):
        """Register a getter function

        Args:
            keys (int or string or tuple of strings): The number or name(s) of
                data that the getter function returns.
            getter (callable): A getter function that takes an index and
                returns data of the corresponding example.
        """
        self._getters.append(getter)
        if isinstance(keys, int):
            if keys == 1:
                keys = None
            else:
                keys = (None,) * keys
        if isinstance(keys, tuple):
            for key_index, key in enumerate(keys):
                self._keys.append((key, len(self._getters) - 1, key_index))
        else:
            self._keys.append((keys, len(self._getters) - 1, None))

    def get_example_by_keys(self, index, key_indices):
        example = []
        cache = {}
        for key_index in key_indices:
            _, getter_index, key_index = self._keys[key_index]
            if getter_index not in cache:
                cache[getter_index] = self._getters[getter_index](index)
            if key_index is None:
                example.append(cache[getter_index])
            else:
                example.append(cache[getter_index][key_index])
        return tuple(example)
