import numpy
import six

from chainer.dataset.indexer import BaseFeaturesIndexer


class TupleDataset(object):

    """Dataset of tuples from multiple equal-length datasets.

    A ``TupleDataset`` combines multiple equal-length datasets into a single
    dataset of tuples. The ``i``-th tuple contains the ``i``-th example from
    each of the argument datasets, in the same order that they were supplied.

    Recall that in Chainer, a dataset is defined as an iterable that supports
    both ``__getitem__`` and ``__len__``. The ``__getitem__`` method should
    support indexing by both an integer and a slice.

    As an example, consider creating a ``TupleDataset`` from two argument
    datasets ``d1 = [8, 0, 5, 1]`` and ``d2 = [3, 1, 7, 4]`` as
    ``tuple_dataset = TupleDataset(d1, d2)``. The ``tuple_dataset`` will
    then contain the examples ``(8, 3), (0, 1), (5, 7), (1, 4)``. Note that
    this behavior is similar to that of the built-in :func:`zip` function.

    Args:
        datasets: Underlying datasets that will be aggregated. Each dataset
            must be an iterable that implements ``__getitem__`` and
            ``__len__``. The ``j``-th dataset will be used for the ``j``-th
            item of each example tuple. All datasets must have the same length.

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
        self._features_indexer = TupleDatasetFeaturesIndexer(self)

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


class TupleDatasetFeaturesIndexer(BaseFeaturesIndexer):
    """FeaturesIndexer for TupleDataset"""

    def __init__(self, dataset):
        """
        
        Args:
            dataset (TupleDataset): TupleDataset instance
        """
        super(TupleDatasetFeaturesIndexer, self).__init__(dataset)
        self.datasets = dataset._datasets

    @property
    def features_length(self):
        return len(self.datasets)

    def extract_feature_by_slice(self, slice_index, j):
        return self.datasets[j][slice_index]

    def extract_feature(self, i, j):
        return self.datasets[j][i]
