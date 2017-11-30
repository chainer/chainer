from chainer.datasets.sliceable import SliceableGetterDataset


class TransformedDataset(SliceableGetterDataset):

    def __init__(self, base, func, keys):
        super(TransformedDataset, self).__init__()
        self._base = base
        self.add_getter(lambda i: func(base.get_example(i)), keys)

    def __len__(self):
        return len(self._base)


def transform(dataset, func, keys):
    return TransformedDataset(dataset, func, keys)
