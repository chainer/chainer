import numpy
import six

from chainer.dataset.indexer import BaseFeaturesIndexer, \
    ExtractBySliceNotSupportedError


class DatasetMixin(object):

    """Default implementation of dataset indexing.

    DatasetMixin provides the :meth:`__getitem__` operator. The default
    implementation uses :meth:`get_example` to extract each example, and
    combines the results into a list. This mixin makes it easy to implement a
    new dataset that does not support efficient slicing.

    Dataset implementation using DatasetMixin still has to provide the
    :meth:`__len__` operator explicitly.

    """
    _features_indexer = None

    def __getitem__(self, index):
        """Returns an example or a sequence of examples.

        It implements the standard Python indexing and one-dimensional integer
        array indexing. It uses the :meth:`get_example` method by default, but
        it may be overridden by the implementation to, for example, improve the
        slicing performance.

        Args:
            index (int, slice, list or numpy.ndarray): An index of an example
                or indexes of examples.

        Returns:
            If index is int, returns an example created by `get_example`.
            If index is either slice or one-dimensional list or numpy.ndarray,
            returns a list of examples created by `get_example`.

        .. admonition:: Example

           >>> import numpy
           >>> from chainer import dataset
           >>> class SimpleDataset(dataset.DatasetMixin):
           ...     def __init__(self, values):
           ...         self.values = values
           ...     def __len__(self):
           ...         return len(self.values)
           ...     def get_example(self, i):
           ...         return self.values[i]
           ...
           >>> ds = SimpleDataset([0, 1, 2, 3, 4, 5])
           >>> ds[1]   # Access by int
           1
           >>> ds[1:3]  # Access by slice
           [1, 2]
           >>> ds[[4, 0]]  # Access by one-dimensional integer list
           [4, 0]
           >>> index = numpy.arange(3)
           >>> ds[index]  # Access by one-dimensional integer numpy.ndarray
           [0, 1, 2]

        """
        if isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return [self.get_example(i) for i in
                    six.moves.range(current, stop, step)]
        elif isinstance(index, list) or isinstance(index, numpy.ndarray):
            return [self.get_example(i) for i in index]
        else:
            return self.get_example(index)

    def __len__(self):
        """Returns the number of data points."""
        raise NotImplementedError

    def get_example(self, i):
        """Returns the i-th example.

        Implementations should override it. It should raise :class:`IndexError`
        if the index is invalid.

        Args:
            i (int): The index of the example.

        Returns:
            The i-th example.

        """
        raise NotImplementedError

    @property
    def features_length(self):
        """Feature size
        
        It should return the number of variables returned by `get_example`.

        """
        raise NotImplementedError

    def extract_feature_by_slice(self, slice_index, j):
        """This method may be override to support efficient feature extraction.
        
        If not override, `extract_feature` is used instead.

        """
        raise ExtractBySliceNotSupportedError

    def extract_feature(self, i, j):
        """Extracts `i`-th data's `j`-th feature
        
        This method may be override to support efficient feature extraction.

        Args:
            i (int): `i`-th data to be extracted
            j (int): `j`-th feature to be extracted

        Returns: feature

        """
        return self.get_example(i)[j]

    @property
    def features(self):
        if self._features_indexer is None:
            self._features_indexer = DatasetMixinFeaturesIndexer(self)
        return self._features_indexer


class DatasetMixinFeaturesIndexer(BaseFeaturesIndexer):
    """FeaturesIndexer for DatasetMixin"""

    def __init__(self, dataset):
        """

        Args:
            dataset (DatasetMixin): DatasetMixin instance
        """
        super(DatasetMixinFeaturesIndexer, self).__init__(dataset)
        self.feature_cache = None

    @property
    def features_length(self):
        return self.dataset.features_length

    def extract_feature_by_slice(self, slice_index, j):
        return self.dataset.extract_feature_by_slice(slice_index, j)

    def extract_feature(self, i, j):
        return self.dataset.extract_feature(i, j)
