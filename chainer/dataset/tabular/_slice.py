import numbers

import numpy as np
import six

from chainer.dataset.tabular import tabular_dataset


class _Slice(tabular_dataset.TabularDataset):

    def __init__(self, dataset, indices, keys):
        self._dataset = dataset
        self._indices = _as_indices(indices, len(dataset))
        self._key_indices = _as_key_indices(keys, dataset.keys)

    def __len__(self):
        if self._indices is None:
            return len(self._dataset)
        elif isinstance(self._indices, slice):
            start, stop, step = self._indices.indices(len(self._dataset))
            return len(six.moves.range(start, stop, step))
        else:
            return len(self._indices)

    @property
    def keys(self):
        if self._key_indices is None:
            return self._dataset.keys
        else:
            return tuple(self._dataset.keys[key_index]
                         for key_index in self._key_indices)

    @property
    def mode(self):
        return self._dataset.mode

    def get_examples(self, indices, key_indices):
        indices = _merge_indices(
            self._indices, indices, len(self._dataset), len(self))
        key_indices = _merge_key_indices(self._key_indices, key_indices)
        return self._dataset.get_examples(indices, key_indices)


class _SliceHelper(object):

    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, args):
        if isinstance(args, tuple):
            indices, keys = args
        else:
            indices = args
            keys = None

        return _Slice(self._dataset, indices, keys)


def _as_indices(indices, len_):
    if isinstance(indices, slice) or len(indices) == 0:
        return indices

    if isinstance(indices[0], (bool, np.bool_)):
        if not len(indices) == len_:
            raise ValueError('The number of booleans is '
                             'different from the length of dataset')
        new_indices = []
        for i, index in enumerate(indices):
            if not isinstance(index, (bool, np.bool_)):
                raise ValueError('{} is not a boolean'.format(index))
            elif index:
                new_indices.append(i)
        return new_indices
    elif isinstance(indices[0], numbers.Integral):
        new_indices = []
        for index in indices:
            if isinstance(index, numbers.Integral):
                if index < 0:
                    index += len_
                if index < 0 or len_ <= index:
                    raise IndexError(
                        'index {} is out of bounds ''for dataset with size {}'
                        .format(index, len_))
            else:
                raise ValueError('{} is not an integer'.format(index))
            new_indices.append(index)
        return new_indices


def _as_key_indices(keys, key_names):
    if keys is None:
        return keys

    key_indices = []
    for key in keys:
        if isinstance(key, numbers.Integral):
            key_index = key
            if key_index < 0:
                key_index += len(key_names)
            if key_index < 0 or len(key_names) <= key_index:
                raise IndexError(
                    'index {} is out of bounds for keys with size {}'.format(
                        key, len(key_names)))
        else:
            try:
                key_index = key_names.index(key)
            except ValueError:
                raise KeyError('{} does not exists'.format(key))
        key_indices.append(key_index)
    return tuple(key_indices)


def _merge_indices(a, b, len_a, len_b):
    if a is None and b is None:
        return None
    elif a is None:
        return b
    elif b is None:
        return a
    elif isinstance(a, slice) and isinstance(b, slice):
        a_start, a_stop, a_step = a.indices(len_a)
        b_start, b_stop, b_step = b.indices(len_b)

        start = a_start + a_step * b_start
        stop = a_start + a_step * b_stop
        step = a_step * b_step

        if start < 0 and step > 0:
            start = None
        if stop < 0 and step < 0:
            stop = None

        return slice(start, stop, step)
    elif isinstance(a, slice):
        a_start, _, a_step = a.indices(len_a)
        return [a_start + a_step * index for index in b]
    elif isinstance(b, slice):
        return a[b]
    else:
        return [a[index] for index in b]


def _merge_key_indices(a, b):
    if a is None and b is None:
        return None
    elif a is None:
        return b
    elif b is None:
        return a
    else:
        return tuple(a[index] for index in b)
