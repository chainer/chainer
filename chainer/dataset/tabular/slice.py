import numbers
import numpy as np

from chainer.dataset import TabularDataset


class Slice(TabularDataset):

    def __init__(self, dataset, indices, keys):
        self._dataset = dataset
        self._indices = _as_indices(indices, len(dataset))
        self._key_indices = _as_key_indices(keys, dataset.keys)

    def __len__(self):
        if self._indices is None:
            return len(self._dataset)
        elif isinstance(self._indices, slice):
            return len(range(
                self._indices.start, self._indices.stop, self._indices.step))
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
        indices = _merge_indices(self._indices, indices)
        key_indices = _merge_key_indices(self._key_indices, key_indices)
        return self._dataset.get_examples(indices, key_indices)


class SliceHelper(object):

    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, args):
        if isinstance(args, tuple):
            indices, keys = args
        else:
            indices = args
            keys = None

        return Slice(self._dataset, indices, keys)


def _as_indices(indices, len_):
    if isinstance(indices, slice):
        start, stop, step = indices.indices(len_)
        return slice(start, stop, step)

    if len(indices) == 0:
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


def _merge_indices(a, b):
    if a is None or b is None:
        return a or b
    elif isinstance(a, slice) and isinstance(b, slice):
        return slice(
            a.start + a.step * b.start,
            a.start + a.step * b.stop,
            a.step * b.step)
    elif isinstance(a, slice):
        return [a.start + a.step * index for index in b]
    elif isinstance(b, slice):
        return [a[index] for index in range(b.start, b.stop, b.step)]
    else:
        return [a[index] for index in b]


def _merge_key_indices(a, b):
    if a is None or b is None:
        return a or b
    else:
        return tuple(a[index] for index in b)
