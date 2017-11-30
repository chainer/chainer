import six

from chainer.datasets.sliceable import SliceableDataset


def _as_tuple(t):
    if isinstance(t, tuple):
        return t
    else:
        return t,


class TupleDataset(SliceableDataset):

    def __init__(self, *datasets):
        if len(datasets) == 0:
            raise ValueError('At least one dataset is required')
        self._len = len(datasets[0])
        self._keys = []
        self._sliceable_datasets = []
        self._regular_datasets = []
        for dataset in datasets:
            if isinstance(dataset, SliceableDataset):
                if not len(dataset) == self._len:
                    raise ValueError(
                        'All datasets should have the same length')
                self._keys.extend(_as_tuple(dataset.keys))
                self._sliceable_datasets.append((set(dataset.keys), dataset))
            else:
                key, dataset = dataset
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

    def get_example_by_keys(self, i, keys):
        required_keys = set(keys)
        values = {}

        for available_keys, dataset in self._sliceable_datasets:
            call_keys = tuple(required_keys.intersection(available_keys))
            if len(call_keys) > 0:
                values.update(six.moves.zip(
                    call_keys, dataset.get_example_by_keys(i, call_keys)))

        for key, dataset in self._regular_datasets:
            if key in required_keys:
                values[key] = dataset[i]

        return tuple(values[key] for key in keys)
