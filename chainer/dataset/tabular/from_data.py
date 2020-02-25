import chainer
from chainer.dataset.tabular import tabular_dataset


def from_data(data, *, size=None):
    """Create a TabularDataset from lists/arrays/callables.

    >>> from chainer.dataset import tabular
    >>>
    >>> dataset = tabular.from_data([0, 1, 2])
    >>> dataset[0]
    0
    >>> dataset = tabular.from_data(([0, 1, 2], [3, 4, 5]))
    >>> dataset[0]
    (0, 3)
    >>> dataset = tabular.from_data((('a', [0, 1, 2]), ('b', [3, 4, 5])))
    >>> dataset.keys
    ('a', 'b')
    >>> dataset[0]
    (0, 3)
    >>> dataset = tabular.from_data({'a': [0, 1, 2], 'b': [3, 4, 5]})
    >>> sorted(dataset[0].items())
    [('a', 0), ('b', 3)]
    >>> dataset = tabular.from_data(('a', lambda i: i * i), size=10)
    >>> dataset[5]
    25

    Args:
        data (list, array, tuple, or dict): Data in following format.

            - `list/array`
            - `(str, list/array/callable)`
            - `((str, ...), callable)`
            - `((list/array)/(str, list/array/callable) \
                /((key, ...), callable), ...)`
            - `{str: (list/array/callable)/(str, ...): callable, ...}`
        size (int): The length of the dataset.
            This argument is required \
            when no lists/arrays exist in :obj:`data`.
    Return:
        A :class:`~chainer.dataset.TabularDataset`.
    """

    if isinstance(data, tuple):
        if len(data) == 2:
            key, d = data
            if isinstance(key, str):
                return _make_dataset(key, d, size)
            if isinstance(key, tuple) and all(isinstance(k, str) for k in key):
                return _make_dataset(key, d, size)

        for d in data:
            if isinstance(d, tuple):
                _, d = d

            if size is None:
                try:
                    size = len(d)
                except TypeError:
                    pass

        datasets = []
        for d in data:
            if isinstance(d, tuple):
                key, d = d
            else:
                key = None
            datasets.append(_make_dataset(key, d, size))

        return datasets[0].join(*datasets[1:]).astuple()

    elif isinstance(data, dict):
        for d in data.values():
            if size is None:
                try:
                    size = len(d)
                except TypeError:
                    pass

        datasets = []
        for key, d in data.items():
            datasets.append(_make_dataset(key, d, size))

        return datasets[0].join(*datasets[1:]).asdict()

    else:
        return _make_dataset(None, data, size)


def _make_dataset(key, data, size):
    if isinstance(data, chainer.get_array_types()):
        if key is None:
            key = '_{}'.format(id(data))
        return _Array(key, data)
    elif isinstance(data, list):
        if key is None:
            key = '_{}'.format(id(data))
        return _List(key, data)
    elif callable(data):
        if key is None:
            raise ValueError('key(s) must be specified for callable')
        if size is None:
            raise ValueError('size must be specified for callable')
        return _Index(size).transform(key, data)


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


class _Index(tabular_dataset.TabularDataset):

    def __init__(self, size):
        self._len = size

    def __len__(self):
        return self._len

    @property
    def keys(self):
        return 'index',

    @property
    def mode(self):
        return None

    def get_examples(self, indices, key_indices):
        if indices is None:
            indices = slice(None)
        if isinstance(indices, slice):
            start, stop, step = indices.indices(len(self))
            indices = list(range(start, stop, step))

        if key_indices is None:
            key_indices = 0,

        return (indices,) * len(key_indices)
