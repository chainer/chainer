import chainer
from chainer.dataset.tabular import tabular_dataset


def from_data(*args, **kwargs):
    """Create a :class:`~chainer.dataset.TabularDataset` from lists/arrays.

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
        If :obj:`args` is specifed, :attr:`mode` is :class:`tuple`.
        Otherwise, :attr:`mode` is :class:`dict`.
    """
    datasets = []

    for data in args:
        if isinstance(data, tuple):
            key, data = data
        else:
            key = '_{}'.format(id(data))

        if isinstance(data, chainer.get_array_types()):
            datasets.append(_Array(key, data, tuple))
        elif isinstance(data, list):
            datasets.append(_List(key, data, tuple))

    for key, data in kwargs.items():
        if isinstance(data, chainer.get_array_types()):
            datasets.append(_Array(key, data, dict))
        elif isinstance(data, list):
            datasets.append(_List(key, data, dict))

    if len(datasets) == 0:
        raise ValueError('At least one data must be passed')

    return datasets[0].join(*datasets[1:])


class _Array(tabular_dataset.TabularDataset):

    def __init__(self, key, data, mode):
        self._key = key
        self._data = data
        self._mode = mode

    def __len__(self):
        return len(self._data)

    @property
    def keys(self):
        return self._key,

    @property
    def mode(self):
        return self._mode

    def get_examples(self, indices, key_indices):
        if key_indices is None:
            key_indices = 0,

        if indices is None:
            return (self._data,) * len(key_indices)
        else:
            return (self._data[indices],) * len(key_indices)


class _List(tabular_dataset.TabularDataset):

    def __init__(self, key, data, mode):
        self._key = key
        self._data = data
        self._mode = mode

    def __len__(self):
        return len(self._data)

    @property
    def keys(self):
        return self._key,

    @property
    def mode(self):
        return self._mode

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
