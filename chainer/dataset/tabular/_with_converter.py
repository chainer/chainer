from chainer.dataset.tabular import tabular_dataset


class _WithConverter(tabular_dataset.TabularDataset):

    def __init__(self, dataset, converter):
        self._dataset = dataset
        self._converter = converter

    def __len__(self):
        return len(self._dataset)

    @property
    def keys(self):
        return self._dataset.keys

    @property
    def mode(self):
        return self._dataset.mode

    def get_examples(self, indices, key_indices):
        return self._dataset.get_examples(indices, key_indices)

    def convert(self, data):
        if isinstance(data, tuple):
            return self._converter(*data)
        elif isinstance(data, dict):
            return self._converter(**data)
        else:
            return self._converter(data)
