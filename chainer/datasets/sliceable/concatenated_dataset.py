from chainer.datasets.sliceable import SliceableDataset


class ConcatenatedDataset(SliceableDataset):
    """A sliceable version of :class:`chainer.datasets.ConcatenatedDataset`.

    Args:
        datasets: The underlying datasets.
            Each dataset should inherit
            :class:~chainer.datasets.sliceable.Sliceabledataset`.
            and should have the same keys.
    """

    def __init__(self, *datasets):
        if len(datasets) == 0:
            raise ValueError('At least one dataset is required')
        self._datasets = datasets
        self._keys = datasets[0].keys
        for dataset in datasets[1:]:
            if not dataset.keys == self._keys:
                raise ValueError('All datasets should have the same keys')

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    @property
    def keys(self):
        return self._keys

    def get_example_by_keys(self, index, keys):
        if index < 0:
            raise IndexError
        for dataset in self._datasets:
            if index < len(dataset):
                return dataset.get_example_by_keys(index, keys)
            index -= len(dataset)
        raise IndexError
