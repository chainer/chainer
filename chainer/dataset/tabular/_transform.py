import six

from chainer.dataset.tabular import tabular_dataset


class _Transform(tabular_dataset.TabularDataset):

    def __init__(self, dataset, keys, transform):
        if not isinstance(keys, tuple):
            keys = keys,

        self._dataset = dataset
        self._keys = keys
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    @property
    def keys(self):
        return self._keys

    @property
    def mode(self):
        if not hasattr(self, '_mode'):
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
            elif self._dataset.mode is None:
                out_example = self._transform(*in_example)

            if isinstance(out_example, tuple):
                if hasattr(self, '_mode') and self._mode is not tuple:
                    raise ValueError(
                        'transform must not change its return type')
                self._mode = tuple
                for col_index, key_index in enumerate(key_indices):
                    out_examples[col_index].append(out_example[key_index])
            elif isinstance(out_example, dict):
                if hasattr(self, '_mode') and self._mode is not dict:
                    raise ValueError(
                        'transform must not change its return type')
                self._mode = dict
                for col_index, key_index in enumerate(key_indices):
                    out_examples[col_index].append(
                        out_example[self._keys[key_index]])
            else:
                if hasattr(self, '_mode') and self._mode is not None:
                    raise ValueError(
                        'transform must not change its return type')
                self._mode = None
                out_example = out_example,
                for col_index, key_index in enumerate(key_indices):
                    out_examples[col_index].append(out_example[key_index])

        return out_examples

    def convert(self, data):
        return self._dataset.convert(data)


class _TransformBatch(tabular_dataset.TabularDataset):

    def __init__(self, dataset, keys, transform_batch):
        if not isinstance(keys, tuple):
            keys = keys,

        self._dataset = dataset
        self._keys = keys
        self._transform_batch = transform_batch

    def __len__(self):
        return len(self._dataset)

    @property
    def keys(self):
        return self._keys

    @property
    def mode(self):
        if not hasattr(self, '_mode'):
            self.get_examples([0], None)
        return self._mode

    def get_examples(self, indices, key_indices):
        if indices is None:
            len_ = len(self)
        elif isinstance(indices, slice):
            start, stop, step = indices.indices(len(self))
            len_ = len(six.moves.range(start, stop, step))
        else:
            len_ = len(indices)

        if key_indices is None:
            key_indices = six.moves.range(len(self._keys))

        in_examples = self._dataset.get_examples(indices, None)

        if self._dataset.mode is tuple:
            out_examples = self._transform_batch(*in_examples)
        elif self._dataset.mode is dict:
            out_examples = self._transform_batch(
                **dict(six.moves.zip(self._dataset.keys, in_examples)))
        elif self._dataset.mode is None:
            out_examples = self._transform_batch(*in_examples)

        if isinstance(out_examples, tuple):
            if hasattr(self, '_mode') and self._mode is not tuple:
                raise ValueError(
                    'transform_batch must not change its return type')
            self._mode = tuple
            if not all(len(col) == len_ for col in out_examples):
                raise ValueError(
                    'transform_batch must not change the length of data')
            return tuple(out_examples[key_index]
                         for key_index in key_indices)
        elif isinstance(out_examples, dict):
            if hasattr(self, '_mode') and self._mode is not dict:
                raise ValueError(
                    'transform_batch must not change its return type')
            self._mode = dict
            if not all(len(col) == len_ for col in out_examples.values()):
                raise ValueError(
                    'transform_batch must not change the length of data')
            return tuple(out_examples[self._keys[key_index]]
                         for key_index in key_indices)
        else:
            if hasattr(self, '_mode') and self._mode is not None:
                raise ValueError(
                    'transform_batch must not change its return type')
            self._mode = None
            out_examples = out_examples,
            if not all(len(col) == len_ for col in out_examples):
                raise ValueError(
                    'transform_batch must not change the length of data')
            return tuple(out_examples[key_index]
                         for key_index in key_indices)

    def convert(self, data):
        return self._dataset.convert(data)
