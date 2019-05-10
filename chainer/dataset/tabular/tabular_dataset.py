class TabularDataset(object):

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

    def __getitem__(self, index):
        slice_ = self.slice[index]
        if not isinstance(slice_, TabularDataset):
            return slice_

        examples = slice_.get_examples(None, None)
        if slice_.mode is tuple:
            return list(zip(*examples))
        elif slice_.mode is dict:
            return [dict(zip(self.keys, example))
                    for example in zip(*examples)]

    def as_tuple(self):
        from chainer.dataset.tabular.as_mode import AsTuple
        return AsTuple(self)

    def as_dict(self):
        from chainer.dataset.tabular.as_mode import AsDict
        return AsDict(self)
