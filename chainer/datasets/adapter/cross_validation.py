import random

import six

from chainer import dataset


class CrossValidationTrainingDataset(dataset.Dataset):

    """Training dataset of one split in cross validation.

    This is a dataset adapter that wraps a base dataset to provide a half of a
    split for cross validation. Together with the corresponding
    :class:`CrossValidationTestDataset` object, it provides one split of the
    base dataset. This dataset represents the larger part of the base dataset.

    Args:
        baseset (Dataset): Base dataset.
        order (sequence of ints): Order of indexes with which each split is
            made from. It is typically a random permutation of
            ``range(len(baseset))``.
        n_fold (int): Number of splits for cross validation.
        index (int): Index of splitting. This dataset represents the training
            set of the ``index``-th split of n-fold cross validation.

    """
    def __init__(self, baseset, order, n_fold, index):
        self._baseset = baseset
        self._order = order
        self._threshold = len(baseset) * index // n_fold
        self._skip = len(baseset) * (index + 1) // n_fold - self._threshold

    def __len__(self):
        return len(self._baseset) - self._skip

    def __getitem__(self, i):
        if i >= self._threshold:
            i += self._skip
        return self._baseset[self._order[i]]


class CrossValidationTestDataset(dataset.Dataset):

    """Test dataset of one split in cross validation.

    This is a dataset adapter that wraps a base dataset to provide a half of a
    split for cross validation. Together with the corresponding
    :class:`CrossValidationTrainingDataset` object, it provides one split of
    the base dataset. This dataset represents the smaller part of the base
    dataset.

    Args:
        baseset (Dataset): Base dataset.
        order (sequence of ints): Order of indexes with which each split is
            determined. It is typically a random permutation of
            ``range(len(baseset))``.
        n_fold (int): Number of splits for cross validation.
        index (int): Index of splitting. This dataset represents the test set
            of the ``index``-th split of n-fold cross validation.

    """
    def __init__(self, baseset, order, n_fold, index):
        self._baseset = baseset
        self._order = order
        self._start = len(baseset) * index // n_fold
        self._size = len(baseset) * (index + 1) // n_fold - self._start

    def __len__(self):
        return self._size

    def __getitem__(self, i):
        return self._baseset[self._order[self._start + i]]


def get_cross_validation_datasets(dataset, n_fold, order=None):
    """Creates a set of training/test dataset pairs for cross validation.

    This function generates ``n_fold`` splits of the base dataset.

    Args:
        dataset (Dataset): Base dataset.
        n_fold (int): Number of splits for cross validation.
        order (sequence of ints): Order of indexes with which each split is
            determined. If it is None, then a random permutation is used.

    Returns:
        list of tuples: List of split datasets. Each entry is a tuple of
            a :class:`CrossValidationTrainingDataset` object and
            corresponding :class:`CrossValidationTestDataset` object.

    """
    if order is None:
        order = list(six.moves.range(len(dataset)))
    random.shuffle(order)
    return [(CrossValidationTrainingDataset(dataset, order, n_fold, i),
             CrossValidationTestDataset(dataset, order, n_fold, i))
            for i in range(n_fold)]
