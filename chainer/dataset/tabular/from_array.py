from chainer.dataset.tabular import tabular_dataset


def from_array(key, array, mode=tuple):
    if mode not in {tuple, dict}:
        raise ValueError('Unknown mode {}'.format(mode))
    return Array(key, array, mode)


class Array(tabular_dataset.TabularDataset):

    def __init__(self, key, array, mode):
        self._key = key
        self._array = array
        self._mode = mode

    def __len__(self):
        return len(self._array)

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
            return (self._array,) * len(key_indices)
        else:
            return (self._array[indices],) * len(key_indices)
