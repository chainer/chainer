import random

import six

from chainer import dataset


class SubDataset(dataset.Dataset):

    """Subset of a base dataset.

    TODO(beam2d): document it.

    """
    def __init__(self, baseset, start, finish, order=None):
        if start < 0 or finish > len(baseset):
            raise ValueError('subset overruns the base dataset')
        self._baseset = baseset
        self._start = start
        self._size = finish - start
        self._order = order

    def __len__(self):
        return self._size

    def __getitem__(self, i):
        index = self._start + i
        if self._order is not None:
            index = self._order[index]
        return self._baseset[index]


def split_dataset(dataset, split_at, order=None):
    subset1 = SubDataset(dataset, 0, split_at, order)
    subset2 = SubDataset(dataset, split_at, len(dataset), order)
    return subset1, subset2


def split_dataset_random(dataset, split_at):
    order = list(six.moves.range(len(dataset)))
    random.shuffle(order)
    return split_dataset(dataset, split_at, order)
