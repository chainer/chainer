from collections import defaultdict
import six

from chainer.datasets.sliceable import SliceableDataset


def _as_tuple(t):
    if isinstance(t, tuple):
        return t
    else:
        return t,


class TupleDataset(SliceableDataset):
    """A sliceable version of :class:`chainer.datasets.TupleDataset`.

    Args:
        datasets: The underlying datasets.
            Following datasets are acceptable.

            * An inheritance of \
                :class:~chainer.datasets.sliceable.SliceableDataset`.
            * A tuple of a name and a data array. \
                 The data array should be list/tuple/:class:`numpy.ndarray`.
            * A data array. In this case, the name of key is :obj:`None`.
    """

    def __init__(self, *datasets):
        if len(datasets) == 0:
            raise ValueError('At least one dataset is required')
        self._len = None
        self._keys = []
        self._datasets = []
        for dataset in datasets:
            if isinstance(dataset, SliceableDataset):
                self._datasets.append(dataset)
                for key_index, key in enumerate(_as_tuple(dataset.keys)):
                    self._keys.append(
                        (key, len(self._datasets) - 1, key_index))
            else:
                if isinstance(dataset, tuple):
                    key, dataset = dataset
                else:
                    key = None
                self._datasets.append(dataset)
                self._keys.append((key, len(self._datasets) - 1, None))
            if self._len is None:
                self._len = len(dataset)
            if not len(dataset) == self._len:
                raise ValueError(
                    'All datasets should have the same length')

    def __len__(self):
        return self._len

    @property
    def keys(self):
        return tuple(key for key, _, _ in self._keys)

    def get_example_by_keys(self, index, key_indices):
        datasets_key_indices = defaultdict(set)
        for key_index in key_indices:
            _, dataset_index, key_index = self._keys[key_index]
            if key_index is None:
                datasets_key_indices[dataset_index] = None
            else:
                datasets_key_indices[dataset_index].add(key_index)

        values = {}
        for dataset_index, dataset_key_indices in \
                six.iteritems(datasets_key_indices):
            dataset = self._datasets[dataset_index]
            if dataset_key_indices is None:
                values[(dataset_index, None)] = dataset[index]
            else:
                dataset_key_indices = tuple(dataset_key_indices)
                values.update(six.moves.zip(
                    ((dataset_index, key_index)
                     for key_index in dataset_key_indices),
                    dataset.get_example_by_keys(index, dataset_key_indices)))

        return tuple(
            values[self._keys[key_index][1:]] for key_index in key_indices)
