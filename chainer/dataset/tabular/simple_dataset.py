import chainer
from chainer.dataset.tabular import _wrappers
from chainer.dataset.tabular import tabular_dataset


class SimpleDataset(tabular_dataset.TabularDataset):

    """A helper class to implement a TabularDataset.

    This class is designed to implement a TabularDataset easily.
    An inheritance of this class can add its columns by :meth:`add_column`.
    A column can be an array or a list or a callable.
    If an array or a list is specified, :meth:`__len__` is implementaed
    automatically by using the length of the given data.

    >>> import numpy as np
    >>>
    >>> from chainer.dataset import tabular
    >>>
    >>> class MyDataset(tabular.SimpleDataset):
    ...
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.add_column('a', np.arange(10))
    ...         self.add_column('b', self.get_b)
    ...         self.add_column('c', [3, 1, 4, 5, 9, 2, 6, 8, 7, 0])
    ...         self.add_column(('d', 'e'), self.get_de)
    ...
    ...     def get_b(self, i):
    ...         return 'b[{}]'.format(i)
    ...
    ...     def get_de(self, i):
    ...         return {'d': 'd[{}]'.format(i), 'e': 'e[{}]'.format(i)}
    ...
    >>> dataset = MyDataset()
    >>> len(dataset)
    10
    >>> dataset.keys
    ('a', 'b', 'c', 'd', 'e')
    >>> dataset[0]
    (0, 'b[0]', 3, 'd[0]', 'e[0]')
    """

    def __init__(self):
        self._len = None
        self._mode = tuple
        self._columns = []
        self._dataset = None

    def add_column(self, key, value):
        """Register column data or callable.

        Args:
            key (str or tuple of strs): The name(s) of key(s).
            value (array or list or callable): The data of column(s).
        """
        self._columns.append((key, value))
        self._dataset = None

    def __len__(self):
        if self._len is None:
            for _, value in self._columns:
                if isinstance(value, chainer.get_array_types() + (list,)):
                    self._len = len(value)
                    break
        if self._len is not None:
            return self._len
        else:
            raise NotImplementedError

    def _get_dataset(self):
        if self._dataset is None:
            datasets = []
            for key, value in self._columns:
                if isinstance(value, chainer.get_array_types()):
                    datasets.append(_wrappers._Array(key, value, tuple))
                elif isinstance(value, list):
                    datasets.append(_wrappers._List(key, value, tuple))
                elif callable(value):
                    datasets.append(_Index(len(self)).transform(key, value))
            self._dataset = datasets[0].join(*datasets[1:])

            if not len(self._dataset) == len(self):
                raise ValueError(
                    'The length of lists/arrays must be same as __len__')
        return self._dataset

    @property
    def keys(self):
        return self._get_dataset().keys

    @keys.setter
    def keys(self, keys):
        self._dataset = self._get_dataset().slice[:, keys]

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if mode not in {tuple, dict, None}:
            raise ValueError('Unknown mode: {}'.format(mode))
        if mode is None and len(self._columns) > 1:
            raise ValueError(
                'Unary mode does not work with multiple columns')
        self._mode = mode

    def get_examples(self, indices, key_indices):
        return self._get_dataset().get_examples(indices, key_indices)


class _Index(tabular_dataset.TabularDataset):

    def __init__(self, len_):
        self._len = len_

    def __len__(self):
        return self._len

    @property
    def keys(self):
        return 'index',

    @property
    def mode(self):
        return tuple

    def get_examples(self, indices, key_indices):
        if indices is None:
            indices = slice(None)
        if isinstance(indices, slice):
            start, stop, step = indices.indices(len(self))
            indices = list(range(start, stop, step))

        if key_indices is None:
            key_indices = 0,

        return (indices,) * len(key_indices)
