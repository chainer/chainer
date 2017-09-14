import six

from chainer.dataset.indexer import BaseFeatureIndexer


class TupleDataset(object):

    """Dataset of a tuple of datasets.

    It combines multiple datasets into one dataset. Each example is represented
    by a tuple whose ``i``-th item corresponds to the i-th dataset.

    Args:
        datasets: Underlying datasets. The ``i``-th one is used for the
            ``i``-th item of each example. All datasets must have the same
            length.

    """

    def __init__(self, *datasets):
        if not datasets:
            raise ValueError('no datasets are given')
        length = len(datasets[0])
        for i, dataset in enumerate(datasets):
            if len(dataset) != length:
                raise ValueError(
                    'dataset of the index {} has a wrong length'.format(i))
        self._datasets = datasets
        self._length = length
        self._features_indexer = TupleDatasetFeatureIndexer(self)

    def __getitem__(self, index):
        batches = [dataset[index] for dataset in self._datasets]
        if isinstance(index, slice):
            length = len(batches[0])
            return [tuple([batch[i] for batch in batches])
                    for i in six.moves.range(length)]
        else:
            return tuple(batches)

    def __len__(self):
        return self._length

    @property
    def features(self):
        """Extract features according to the specified index.

        - axis 0 is used to specify dataset id (`i`-th dataset)
        - axis 1 is used to specify feature index

        .. admonition:: Example

           >>> from chainer.datasets import TupleDataset
           >>> tuple_dataset = TupleDataset([0, 1, 2], [0, 1, 4])
           >>> targets = tuple_dataset.features[:, 1]
           >>> print('targets', targets)  # We can extract only target value
           targets [0, 1, 4]

        """
        return self._features_indexer


class TupleDatasetFeatureIndexer(BaseFeatureIndexer):
    """FeatureIndexer for TupleDataset"""

    def __init__(self, dataset):
        super(TupleDatasetFeatureIndexer, self).__init__(dataset)
        self.datasets = dataset._datasets

    @property
    def features_length(self):
        return len(self.datasets)

    def extract_feature_by_slice(self, slice_index, j):
        return self.datasets[j][slice_index]

    def extract_feature(self, i, j):
        return self.datasets[j][i]
