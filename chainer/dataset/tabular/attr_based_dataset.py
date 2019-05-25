import contextlib

import chainer
from chainer.dataset.tabular import _wrappers
from chainer.dataset.tabular import tabular_dataset


class AttrBasedDataset(tabular_dataset.TabularDataset):

    def __init__(self):
        self._within_init_scope = False
        self._len = None
        self._mode = tuple
        self._data = []
        self._dataset = None

    def _get_dataset(self):
        if self._dataset is None:
            datasets = []
            for key, value in self._data:
                if isinstance(value, chainer.get_array_types()):
                    datasets.append(_wrappers._Array(key, value, tuple))
                elif isinstance(value, list):
                    datasets.append(_wrappers._List(key, value, tuple))
                elif callable(value):
                    datasets.append(_Index(len(self)).transform(key, value))
            self._dataset = datasets[0].join(*datasets[1:])
        return self._dataset

    def __len__(self):
        if self._len is None:
            for _, value in self._data:
                if isinstance(value, chainer.get_array_types() + (list,)):
                    self._len = len(value)
                    break
        if self._len is not None:
            return self._len
        else:
            raise NotImplementedError

    @property
    def keys(self):
        return self._get_dataset().keys

    @property
    def mode(self):
        return self._mode

    def get_examples(self, indices, key_indices):
        return self._get_dataset().get_examples(indices, key_indices)

    @contextlib.contextmanager
    def init_scope(self):
        self._dataset = None
        prev, self._within_init_scope = self._within_init_scope, True
        try:
            yield
        finally:
            self._within_init_scope = prev

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        self[key] = value

    def __setitem__(self, key, value):
        if self._within_init_scope:
            if isinstance(value, chainer.get_array_types() + (list,)) or \
               callable(value):
                self._data.append((key, value))


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
