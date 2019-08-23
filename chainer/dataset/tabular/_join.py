import six

from chainer.dataset.tabular import tabular_dataset


class _Join(tabular_dataset.TabularDataset):

    def __init__(self, *datasets):
        keys = set(datasets[0].keys)
        for dataset in datasets[1:]:
            if not len(dataset) == len(datasets[0]):
                raise ValueError('All datasets must have the same length')
            if len(keys.intersection(dataset.keys)) > 0:
                raise ValueError('All keys must be unique among all datasets')
            keys = keys.union(dataset.keys)

        self._datasets = datasets

    def __len__(self):
        return len(self._datasets[0])

    @property
    def keys(self):
        return tuple(key for dataset in self._datasets for key in dataset.keys)

    @property
    def mode(self):
        for dataset in self._datasets:
            if dataset.mode:
                return dataset.mode
        return tuple

    def get_examples(self, indices, key_indices):
        if key_indices is None:
            return tuple(
                col
                for dataset in self._datasets
                for col in dataset.get_examples(indices, None))

        examples = {}
        key_offset = 0
        for dataset in self._datasets:
            sub_key_indices = []
            for key_index in key_indices:
                sub_key_index = key_index - key_offset
                if sub_key_index < 0 or len(dataset.keys) <= sub_key_index:
                    continue
                if sub_key_index not in sub_key_indices:
                    sub_key_indices.append(sub_key_index)

            if len(sub_key_indices) > 0:
                sub_key_indices = tuple(sub_key_indices)
                sub_examples = dataset.get_examples(indices, sub_key_indices)
                for sub_key_index, col_example in six.moves.zip(
                        sub_key_indices, sub_examples):
                    examples[key_offset + sub_key_index] = col_example

            key_offset += len(dataset.keys)

        return tuple(examples[key_index] for key_index in key_indices)

    def convert(self, data):
        return self._datasets[0].convert(data)
