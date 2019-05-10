from chainer.dataset import TabularDataset


class View(TabularDataset):

    def __init__(self, dataset, indices, key_indices, mode):
        self._dataset = dataset
        self._indices = indices
        self._key_indices = key_indices
        self._mode = mode

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
        return self._mode

    def get_examples(self, indices, key_indices):
        indices = _merge_indices(self._indices, indices)
        key_indices = _merge_key_indices(self._key_indices, key_indices)
        return self._dataset.get_examples(indices, key_indices)


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
