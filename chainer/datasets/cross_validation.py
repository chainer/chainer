import random

import six

from chainer import dataset


class CrossValidationTrainingDataset(dataset.Dataset):

    """Training dataset for cross validation.

    TODO(beam2d): document it.

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

    """Test dataset for cross validation.

    TODO(beam2d): document it.

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


def get_cross_validation_datasets(dataset, n_fold):
    order = list(six.moves.range(len(dataset)))
    random.shuffle(order)
    return [(CrossValidationTrainingDataset(dataset, order, n_fold, i),
             CrossValidationTestDataset(dataset, order, n_fold, i))
            for i in range(n_fold)]
