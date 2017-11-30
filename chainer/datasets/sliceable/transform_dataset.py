from chainer.datasets.sliceable import GetterDataset


class TransformDataset(GetterDataset):

    def __init__(self, dataset, func, keys):
        super(TransformDataset, self).__init__()
        self._dataset = dataset
        self.add_getter(lambda i: func(dataset.get_example(i)), keys)

    def __len__(self):
        return len(self._dataset)
