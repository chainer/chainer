import random

import six

from chainer import dataset


class SubDataset(dataset.Dataset):

    """Subset of a base dataset.

    This dataset adapter defines a subset of a given base dataset. The subset
    is defined as an interval of indexes, optionally under a given permutation.

    If ``order`` is given, then the ``i``-th data point of this dataset is the
    ``order[start + i]``-th data point of the base dataset for any non-negative
    integer ``i``. If ``order`` is not given, then the ``i``-th data point of
    this dataset is the ``start + i``-th data point of the base dataset for any
    non-negative integer ``i``. Negative indexing is also allowed: in this
    case, the term ``start + i`` is replaced by ``finish + i``.

    Args:
        baseset (Dataset): Base dataset.
        start (int): Leftmost index of the subset in the base dataset.
        finish (int): Rightmost index of the subset in the base dataset.
        order (sequence of ints): Permutation of indexes in the base dataset.
            If this is None, then identity mapping is used.

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
        if i >= 0:
            if i >= self._size:
                raise IndexError('dataset index out of range')
            index = self._start + i
        else:
            if i < -self._size:
                raise IndexError('dataset index out of range')
            index = self._start + self._size + i

        if self._order is not None:
            index = self._order[index]
        return self._baseset[index]


def split_dataset(dataset, split_at, order=None):
    """Splits a dataset into two subsets.

    This function creates two instances of the :class:`SubDataset` class. These
    instances do not share any data points, and they together cover all data
    points of the original dataset.

    Args:
        dataset (Dataset): Base dataset.
        split_at (int): Position at which the base dataset is split.
        order (sequence of ints): Permutation of indexes in the base dataset.
            See the document of :class:`SubDataset` for details.

    Returns:
        tuple: Two :class:`SubDataset` objects. The first subset represents the
            data points of indexes ``order[:split_at]`` while the second
            subset represents the data points of indexes ``order[split_at:]``.

    """
    subset1 = SubDataset(dataset, 0, split_at, order)
    subset2 = SubDataset(dataset, split_at, len(dataset), order)
    return subset1, subset2


def split_dataset_random(dataset, split_at):
    """Splits a dataset into two subsets randomly.

    This function calls the :func:`split_dataset` function with random
    permutation, thus provides a random separation of the base dataset.

    Args:
        dataset (Dataset): Base dataset.
        split_at (int): The size of the first subset.

    Returns:
        tuple: Two :class:`SubDataset` objects. The first subset is of size
            ``split_at``, and the latter of size ``len(dataset) - split_at``.

    """
    order = list(six.moves.range(len(dataset)))
    random.shuffle(order)
    return split_dataset(dataset, split_at, order)
