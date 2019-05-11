import six

from chainer.dataset import TabularDataset


class Concat(TabularDataset):

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
        if indices is None:
            indices = slice(0, len(self), 1)
        if key_indices is None:
            n_cols = len(self.keys)
        else:
            n_cols = len(key_indices)

        examples = [[] for _ in six.moves.range(n_cols)]
        offset = 0
        for dataset in self._datasets:
            if isinstance(indices, slice):
                start = indices.start - offset
                stop = indices.stop - offset
                step = indices.step
                if step > 0:
                    if start < 0:
                        start %= step
                    stop = min(stop, len(dataset))
                else:
                    if start >= len(dataset):
                        start = len(dataset) + (start - len(dataset)) % step
                    stop = max(stop, -1)
                if len(six.moves.range(start, stop, step)) > 0:
                    sub_indices = slice(start, stop, step)
                else:
                    sub_indices = None
            else:
                sub_indices = [
                    index - offset
                    for index in indices
                    if offset <= index < offset + len(dataset)]
                if len(sub_indices) == 0:
                    sub_indices = None

            if sub_indices is not None:
                sub_examples = dataset.get_examples(sub_indices, key_indices)
                for col_index in six.moves.range(n_cols):
                    examples[col_index].append(sub_examples[col_index])

            offset += len(dataset)

        for col_index in six.moves.range(n_cols):
            if len(examples[col_index]) == 0:
                examples[col_index] = []
            elif len(examples[col_index]) == 1:
                examples[col_index] = examples[col_index][0]
            else:
                examples[col_index] = [
                    example
                    for sub_examples in examples[col_index]
                    for example in sub_examples]

        return tuple(examples)
