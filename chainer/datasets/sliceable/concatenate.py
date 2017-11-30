from chainer.datasets.sliceable import SliceableDataset


class ConcatenatedDataset(SliceableDataset):

    def __init__(self, datasets):
        self._datasets = datasets
        self._keys = datasets[0].keys

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    @property
    def keys(self):
        return self._keys

    def get_example_by_keys(self, i, keys):
        if i < 0:
            raise IndexError
        for dataset in self._datasets:
            if i < len(dataset):
                return dataset.get_example_by_keys(i, keys)
            i -= len(dataset)
        raise IndexError


def concatenate(*datasets):
    if len(datasets) == 0:
        raise ValueError('At least one dataset is required')
    for dataset in datasets[1:]:
        if not dataset.keys == datasets[0].keys:
            raise ValueError('All datasets should have the same keys')
    return ConcatenatedDataset(datasets)
