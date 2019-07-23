import chainer
from chainer.dataset.tabular import tabular_dataset


def from_data(*args, **kwargs):
    """Create a :class:`~chainer.dataset.TabularDataset` from lists/arrays.

    >>> from chainer.dataset import tabular
    >>>
    >>> dataset = tabular.from_data([0, 1, 2])
    >>> dataset[0]
    0
    >>> dataset = tabular.from_data([0, 1, 2], [3, 4, 5])
    >>> dataset[0]
    (0, 3)
    >>> dataset = tabular.from_data(('a', [0, 1, 2]), ('b', [3, 4, 5]))
    >>> dataset.keys
    ('a', 'b')
    >>> dataset[0]
    (0, 3)
    >>> dataset = tabular.from_data(a=[0, 1, 2], b=[3, 4, 5])
    >>> sorted(dataset[0].items())
    [('a', 0), ('b', 3)]

    Args:
        args (list or array or tuple): Data of a column.
            If this argument is an array or a list,
            it is treated as :obj:`data`.
            In this case, the key is generated automatically.
            Do not rely on the key.
            If this argument is a tuple, it is treated as :obj:`(key, data)`.
        kwargs (list or array): Data of a column.
            The order of columns is determined automatically.
            Do not rely on the order.
    Return:
        A :class:`~chainer.dataset.TabularDataset`.
        If only one argument is given, :attr:`mode` is :obj:`None`.
        If more than one arguments are given as :obj:`args`,
        :attr:`mode` is :class:`tuple`.
        If more than one arguments are given as :obj:`kwargs`,
        :attr:`mode` is :class:`dict`.
    """
    datasets = []

    for data in args:
        if isinstance(data, tuple):
            key, data = data
        else:
            key = '_{}'.format(id(data))

        if isinstance(data, chainer.get_array_types()):
            datasets.append(_Array(key, data))
        elif isinstance(data, list):
            datasets.append(_List(key, data))

    for key, data in kwargs.items():
        if isinstance(data, chainer.get_array_types()):
            datasets.append(_Array(key, data))
        elif isinstance(data, list):
            datasets.append(_List(key, data))

    if len(datasets) == 1:
        return datasets[0]
    elif args and kwargs:
        raise ValueError('Mixture of args and kwargs is not supported')
    elif args:
        return datasets[0].join(*datasets[1:]).as_tuple()
    elif kwargs:
        return datasets[0].join(*datasets[1:]).as_dict()
    else:
        raise ValueError('At least one data must be passed')


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
