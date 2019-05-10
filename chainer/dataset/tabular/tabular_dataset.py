import numbers
import numpy as np


class TabularDataset(object):

    def __len__(self):
        raise NotImplementedError

    @property
    def keys(self):
        raise NotImplementedError

    @property
    def mode(self):
        raise NotImplementedError

    def get_examples(self, indices, key_indices):
        raise NotImplementedError

    @property
    def slice(self):
        return SliceHelper(self)

    def fetch(self):
        examples = self.get_examples(None, None)
        if self.mode is tuple:
            return examples
        elif self.mode is dict:
            return dict(zip(self.keys, examples))

    def __getitem__(self, index):
        view = self.slice[index]
        if isinstance(index, numbers.Integral):
            return view

        examples = view.get_examples(None, None)
        if view.mode is tuple:
            return list(zip(*examples))
        elif view.mode is dict:
            return [dict(zip(self.keys, example))
                    for example in zip(*examples)]

    def as_tuple(self):
        from chainer.dataset.tabular.view import View
        return View(self, None, None, tuple)

    def as_dict(self):
        from chainer.dataset.tabular.view import View
        return View(self, None, None, dict)


class SliceHelper(object):

    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, args):
        if isinstance(args, tuple):
            indices, keys = args
        else:
            indices = args
            keys = None

        if isinstance(indices, numbers.Integral):
            single_index = True
            indices = [indices]
        else:
            single_index = False

        indices = _as_indices(indices, len(self._dataset))
        key_indices = _as_key_indices(keys, self._dataset.keys)

        from chainer.dataset.tabular.view import View
        view = View(self._dataset, indices, key_indices, self._dataset.mode)

        if single_index:
            examples = view.get_examples(None, None)
            examples = tuple(col[0] for col in examples)
            if view.mode is tuple:
                return examples
            elif view.mode is dict:
                return dict(zip(view.keys, examples))
        else:
            return view


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
