import six

from chainer.dataset.tabular import tabular_dataset


class _Concat(tabular_dataset.TabularDataset):

    def __init__(self, *datasets):
        for dataset in datasets[1:]:
            if not dataset.keys == datasets[0].keys:
                raise ValueError('All datasets must have the same keys')

        self._datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    @property
    def keys(self):
        return self._datasets[0].keys

    @property
    def mode(self):
        return self._datasets[0].mode

    def get_examples(self, indices, key_indices):
        if key_indices is None:
            n_cols = len(self.keys)
        else:
            n_cols = len(key_indices)

        if indices is None:
            examples = [
                dataset.get_examples(None, key_indices)
                for dataset in self._datasets]
            return tuple(
                [data
                 for sub_examples in examples
                 for data in sub_examples[col_index]]
                for col_index in six.moves.range(n_cols))

        elif isinstance(indices, slice):
            start, stop, step = indices.indices(len(self))

            examples = []
            offset = 0
            for dataset in self._datasets:
                sub_start = start - offset
                sub_stop = stop - offset
                if step > 0:
                    if sub_start < 0:
                        sub_start %= step
                    sub_stop = min(sub_stop, len(dataset))
                else:
                    if sub_start >= len(dataset):
                        sub_start = \
                            len(dataset) + (sub_start - len(dataset)) % step
                    sub_stop = max(sub_stop, -1)

                if len(six.moves.range(sub_start, sub_stop, step)) > 0:
                    if sub_start < 0 and step > 0:
                        sub_start = None
                    if sub_stop < 0 and step < 0:
                        sub_stop = None
                    examples.append(dataset.get_examples(
                        slice(sub_start, sub_stop, step), key_indices))

                offset += len(dataset)

            if len(examples) == 0:
                return tuple([] for _ in six.moves.range(n_cols))
            elif len(examples) == 1:
                return examples[0]
            else:
                if step < 0:
                    examples.reverse()
                return tuple(
                    [data
                     for sub_examples in examples
                     for data in sub_examples[col_index]]
                    for col_index in six.moves.range(n_cols))

        else:
            examples = {}
            example_indices = [None] * len(indices)
            offset = 0
            for dataset_index, dataset in enumerate(self._datasets):
                sub_indices = []
                for p, index in enumerate(indices):
                    if index < offset or offset + len(dataset) <= index:
                        continue
                    sub_indices.append(index - offset)
                    example_indices[p] = (
                        dataset_index, len(sub_indices) - 1)

                if len(sub_indices) > 0:
                    examples[dataset_index] = dataset.get_examples(
                        sub_indices, key_indices)

                offset += len(dataset)

            if len(examples) == 0:
                return tuple([] for _ in six.moves.range(n_cols))
            elif len(examples) == 1:
                return list(examples.values())[0]
            else:
                return tuple(
                    [examples[dataset_index][col_index][p]
                     for dataset_index, p in example_indices]
                    for col_index in six.moves.range(n_cols))

    def convert(self, data):
        return self._datasets[0].convert(data)
