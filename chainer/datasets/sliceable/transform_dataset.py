from chainer.datasets.sliceable import GetterDataset


class TransformDataset(GetterDataset):

    def __init__(self, dataset, keys, func):
        super(TransformDataset, self).__init__()
        self._dataset = dataset
        self.add_getter(keys, lambda i: func(dataset.get_example(i)))

    def __len__(self):
        return len(self._dataset)
