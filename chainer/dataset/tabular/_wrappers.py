from chainer.dataset.tabular import tabular_dataset


class _Array(tabular_dataset.TabularDataset):

    def __init__(self, key, data):
        self._key = key
        self._data = data

    def __len__(self):
        return len(self._data)

    @property
    def keys(self):
        return self._key,

    @property
    def mode(self):
        return None

    def get_examples(self, indices, key_indices):
        if key_indices is None:
            key_indices = 0,

        if indices is None:
            return (self._data,) * len(key_indices)
        else:
            return (self._data[indices],) * len(key_indices)


class _List(tabular_dataset.TabularDataset):

    def __init__(self, key, data):
        self._key = key
        self._data = data

    def __len__(self):
        return len(self._data)

    @property
    def keys(self):
        return self._key,

    @property
    def mode(self):
        return None

    def get_examples(self, indices, key_indices):
        if key_indices is None:
            key_indices = 0,

        if indices is None:
            return (self._data,) * len(key_indices)
        elif isinstance(indices, slice):
            return (self._data[indices],) * len(key_indices)
        else:
            return ([self._data[index] for index in indices],) \
                * len(key_indices)
