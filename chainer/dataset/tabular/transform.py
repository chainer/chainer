import six

from chainer.dataset.tabular import tabular_dataset


class Transform(tabular_dataset.TabularDataset):

    def __init__(self, dataset, keys, transform):
        if keys is not None and not isinstance(keys, tuple):
            keys = keys,

        self._dataset = dataset
        self._keys = keys
        self._mode = None
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    @property
    def keys(self):
        return self._keys

    @property
    def mode(self):
        if self._mode is None:
            self.get_examples([0], None)
        return self._mode

    def get_examples(self, indices, key_indices):
        if key_indices is None:
            key_indices = six.moves.range(len(self._keys))

        in_examples = self._dataset.get_examples(indices, None)

        out_examples = tuple([] for _ in key_indices)
        for in_example in six.moves.zip(*in_examples):
            if self._dataset.mode is tuple:
                out_example = self._transform(*in_example)
            elif self._dataset.mode is dict:
                out_example = self._transform(
                    **dict(six.moves.zip(self._dataset.keys, in_example)))

            if not isinstance(out_example, (tuple, dict)):
                out_example = out_example,
            if isinstance(out_example, tuple):
                if self._keys is None:
                    self._keys = tuple(
                        '_{}'.format(id(out)) for out in out_example)
                self._mode = tuple
                for col_index, key_index in enumerate(key_indices):
                    out_examples[col_index].append(out_example[key_index])
            elif isinstance(out_example, dict):
                if self._keys is None:
                    self._keys = tuple(out_example.keys())
                self._mode = dict
                for col_index, key_index in enumerate(key_indices):
                    out_examples[col_index].append(
                        out_example[self._keys[key_index]])

        return out_examples
