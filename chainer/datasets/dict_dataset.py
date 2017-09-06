import numpy
import six

from chainer.dataset.indexer import BaseFeaturesIndexer


class DictDataset(object):

    """Dataset of a dictionary of datasets.

    It combines multiple datasets into one dataset. Each example is represented
    by a dictionary mapping a key to an example of the corresponding dataset.

    Args:
        datasets: Underlying datasets. The keys are used as the keys of each
            example. All datasets must have the same length.

    """

    def __init__(self, **datasets):
        if not datasets:
            raise ValueError('no datasets are given')
        length = None
        for key, dataset in six.iteritems(datasets):
            if length is None:
                length = len(dataset)
            elif length != len(dataset):
                raise ValueError(
                    'dataset length conflicts at "{}"'.format(key))
        self._datasets = datasets
        self._length = length
        self._features_indexer = DictDatasetFeaturesIndexer(self)

    def __getitem__(self, index):
        batches = {key: dataset[index]
                   for key, dataset in six.iteritems(self._datasets)}
        if isinstance(index, slice):
            length = len(six.itervalues(batches).next())
            return [{key: batch[i] for key, batch in six.iteritems(batches)}
                    for i in six.moves.range(length)]
        else:
            return batches

    def __len__(self):
        return self._length

    @property
    def features(self):
        """Extract features according to the specified index.

        - axis 0 is used to specify dataset id (`i`-th dataset)
        - axis 1 is used to specify feature label

        .. admonition:: Example

           >>> from chainer.datasets import DictDataset
           >>> dd = DictDataset(x=[0, 1, 2], t=[0, 1, 4])
           >>> targets = dd.features[:, 't']
           >>> print('targets', targets)  # We can extract only target value
           targets [0, 1, 4]

        """
        return self._features_indexer


class DictDatasetFeaturesIndexer(BaseFeaturesIndexer):
    """FeaturesIndexer for TupleDataset"""

    def __init__(self, dataset):
        """

        Args:
            dataset (DictDataset): DictDataset instance
        """
        super(DictDatasetFeaturesIndexer, self).__init__(
            dataset, access_feature_by_key=True
        )
        self.datasets = dataset._datasets

    @property
    def features_keys(self):
        return self.datasets.keys()

    @property
    def features_length(self):
        return len(self.datasets)

    def extract_feature_by_slice(self, slice_index, j):
        return self.datasets[j][slice_index]

    def extract_feature(self, i, j):
        return self.datasets[j][i]
