from chainer.dataset import DatasetMixin


class TabularDataset(DatasetMixin):

    def __len__(self):
        raise NotImplementedError

    @property
    def keys(self):
        raise NotImplementedError

    @property
    def mode(self):
        raise NotImplementedError

    def get_examples(self, indices, key_indices):
        raise NotImplementedError

    @property
    def slice(self):
        from chainer.dataset.tabular.slice import SliceHelper
        return SliceHelper(self)

    def fetch(self):
        examples = self.get_examples(None, None)
        if self.mode is tuple:
            return examples
        elif self.mode is dict:
            return dict(zip(self.keys, examples))

    def as_tuple(self):
        from chainer.dataset.tabular.as_mode import AsTuple
        return AsTuple(self)

    def as_dict(self):
        from chainer.dataset.tabular.as_mode import AsDict
        return AsDict(self)

    def get_example(self, index):
        example = self.get_examples([index], None)
        example = tuple(col[0] for col in example)
        if self.mode is tuple:
            return example
        elif self.mode is dict:
            return dict(zip(self.keys, example))
