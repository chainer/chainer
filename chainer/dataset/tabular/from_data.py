import chainer
from chainer.dataset.tabular import tabular_dataset


def from_data(*args, **kwargs):
    datasets = []

    for data in args:
        if isinstance(data, tuple):
            key, data = data
        else:
            key = '_{}'.format(id(data))

        if isinstance(data, chainer.get_array_types()):
            datasets.append(_FromArray(key, data, tuple))
        else:
            datasets.append(_FromList(key, data, tuple))

    for key, data in kwargs.items():
        if isinstance(data, chainer.get_array_types()):
            datasets.append(_FromArray(key, data, dict))
        else:
            datasets.append(_FromList(key, data, dict))

    if len(datasets) == 0:
        raise ValueError('At least one data must be passed')

    return datasets[0].join(*datasets[1:])


class _FromArray(tabular_dataset.TabularDataset):

    def __init__(self, key, data, mode):
        self._key = key
        self._data = data
        self._mode = mode

    def __len__(self):
        return len(self._data)

    @property
    def keys(self):
        return self._key,

    def mode(self):
        return self._mode

    def get_examples(self, indices, key_indices):
        if indices is None:
            return (self._data,) * len(key_indices)
        else:
            return (self._data[indices],) * len(key_indices)


class _FromList(tabular_dataset.TabularDataset):

    def __init__(self, key, data, mode):
        self._key = key
        self._data = data
        self._mode = mode

    def __len__(self):
        return len(self._data)

    @property
    def keys(self):
        return self._key,

    def mode(self):
        return self._mode

    def get_examples(self, indices, key_indices):
        if indices is None:
            return (self._data,) * len(key_indices)
        elif isinstance(indices, slice):
            return (self._data[indices],) * len(key_indices)
        else:
            return ([self._data[index] for index in indices],) \
                * len(key_indices)
