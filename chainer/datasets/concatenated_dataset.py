from chainer.dataset import dataset_mixin


class ConcatenatedDataset(dataset_mixin.DatasetMixin):

    """Dataset which concatenates some base datasets.

    This dataset wraps some base datasets and works as a concatenated dataset.
    For example, if a base dataset with 10 samples and
    another base dataset with 20 samples are given, this dataset works as
    a dataset which has 30 samples.

    Args:
        datasets: The underlying datasets. Each dataset has to support
            :meth:`__len__` and :meth:`__getitem__`.

    """

    def __init__(self, *datasets):
        self._datasets = datasets

    def __len__(self):
        return sum(len(dataset) for dataset in self._datasets)

    def get_example(self, i):
        if i < 0:
            raise IndexError
        for dataset in self._datasets:
            if i < len(dataset):
                return dataset[i]
            i -= len(dataset)
        raise IndexError
