from chainer.datasets.sliceable import GetterDataset


class TransformDataset(GetterDataset):
    """A sliceable version of :class:`chainer.datasets.TransformDataset`.

    Args:
        dataset: The underlying dataset.
            This dataset should have :meth:`__len__` and :meth:`__getitem__`.
        keys (string or tuple of strings): Name(s) of data that
                the transform function returns.
        transform (callable): A function that is called to transform values
            returned by the underlying dataset's :meth:`__getitem__`.
    """

    def __init__(self, dataset, keys, transform):
        super(TransformDataset, self).__init__()
        self._dataset = dataset
        self.add_getter(keys, lambda index: transform(dataset[index]))

    def __len__(self):
        return len(self._dataset)
