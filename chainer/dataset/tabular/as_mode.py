import chainer.dataset


class AsTuple(chainer.dataset.TabularDataset):

    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    @property
    def keys(self):
        return self._dataset.keys

    @property
    def mode(self):
        return tuple

    def get_examples(self, indices, key_indices):
        return self._dataset.get_examples(indices, key_indices)


class AsDict(chainer.dataset.TabularDataset):

    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    @property
    def keys(self):
        return self._dataset.keys

    @property
    def mode(self):
        return dict

    def get_examples(self, indices, key_indices):
        return self._dataset.get_examples(indices, key_indices)
