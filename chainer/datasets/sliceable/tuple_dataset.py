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
            Each dataset should be an inheritance of
            :class:~chainer.datasets.sliceable.Sliceabledataset`.
            or a tuple of the name of datum and a data array
            (list/tuple/:class:`numpy.ndarray`).
    """

    def __init__(self, *datasets):
        if len(datasets) == 0:
            raise ValueError('At least one dataset is required')
        self._len = None
        self._keys = []
        self._sliceable_datasets = []
        self._regular_datasets = []
        for dataset in datasets:
            if isinstance(dataset, SliceableDataset):
                if self._len is None:
                    self._len = len(dataset)
                if not len(dataset) == self._len:
                    raise ValueError(
                        'All datasets should have the same length')
                self._keys.extend(_as_tuple(dataset.keys))
                self._sliceable_datasets.append((set(dataset.keys), dataset))
            else:
                key, dataset = dataset
                if self._len is None:
                    self._len = len(dataset)
                if not len(dataset) == self._len:
                    raise ValueError(
                        'All datasets should have the same length')
                self._keys.append(key)
                self._regular_datasets.append((key, dataset))

    def __len__(self):
        return self._len

    @property
    def keys(self):
        return tuple(self._keys)

    def get_example_by_keys(self, index, keys):
        values = {}

        for available_keys, dataset in self._sliceable_datasets:
            call_keys = tuple(available_keys.intersection(keys))
            if len(call_keys) > 0:
                values.update(six.moves.zip(
                    call_keys, dataset.get_example_by_keys(index, call_keys)))

        for key, dataset in self._regular_datasets:
            if key in keys:
                values[key] = dataset[index]

        return tuple(values[key] for key in keys)
