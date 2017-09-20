import numpy
import six


class ExtractBySliceNotSupportedError(Exception):
    pass


class BaseIndexer(object):
    """Base class for Indexer"""

    def __getitem__(self, item):
        raise NotImplementedError


class BaseFeatureIndexer(BaseIndexer):

    """Base class for FeatureIndexer

    FeatureIndexer can be accessed by 2-dimensional indices, axis=0 is used for
    dataset index and axis=1 is used for feature index.
    For example, let `features` be the instance of `BaseFeatureIndexer`, then
    `features[i, j]` returns `i`-th dataset of `j`-th feature.

    `features[ind]` works same with `features[ind, :]`

    Note that the returned value will be numpy array, even though the
    dataset is initilized with other format (e.g. list).

    """

    def __init__(self, dataset):
        super(BaseFeatureIndexer, self).__init__()
        self.dataset = dataset

    def features_length(self):
        """Returns length of features

        Returns (int): feature length

        """
        raise NotImplementedError

    @property
    def dataset_length(self):
        return len(self.dataset)

    @property
    def shape(self):
        return self.dataset_length, self.features_length()

    def extract_feature_by_slice(self, slice_index, j):
        """Extracts `slice_index`-th data's `j`-th feature.

        Here, `slice_index` is indices of slice object.
        This method may be override to support efficient feature extraction.
        If not override, `ExtractBySliceNotSupportedError` is raised by
        default, and in this case `extract_feature` is used instead.

        Args:
            slice_index (slice): slice of data index to be extracted
            j (int): `j`-th feature to be extracted

        Returns: feature
        """

        raise ExtractBySliceNotSupportedError

    def extract_feature(self, i, j):
        """Extracts `i`-th data's `j`-th feature

        Args:
            i (int): `i`-th data to be extracted
            j (int): `j`-th feature to be extracted

        Returns: feature

        """
        raise NotImplementedError

    def create_feature_index_list(self, feature_index):
        if isinstance(feature_index, slice):
            feature_index_list = numpy.arange(
                *feature_index.indices(self.features_length())
            )
        elif isinstance(feature_index, (list, numpy.ndarray)):
            if isinstance(feature_index[0],
                          (bool, numpy.bool, numpy.bool_)):
                if len(feature_index) != self.features_length():
                    raise ValueError('Feature index wrong length {} instead of'
                                     ' {}'.format(len(feature_index),
                                                  self.features_length()))
                feature_index_list = numpy.argwhere(feature_index
                                                    ).ravel()
            else:
                feature_index_list = feature_index
        else:
            # assuming int type
            feature_index_list = [feature_index]
        return feature_index_list

    def preprocess(self, item):
        pass

    def postprocess(self, item):
        pass

    def __getitem__(self, item):
        self.preprocess(item)
        if isinstance(item, tuple):
            index_dim = len(item)
            # multi dimensional access
            if index_dim == 1:
                # This is not unexpected case...
                data_index = item[0]
                feature_index_list = self.create_feature_index_list(
                    slice(None)
                )
            elif index_dim == 2:
                data_index, feature_index = item
                feature_index_list = self.create_feature_index_list(
                    feature_index
                )
            else:
                raise IndexError('too many indices for features')
        else:
            data_index = item
            feature_index_list = self.create_feature_index_list(slice(None))
        if len(feature_index_list) == 1:
            self._extract_single_feature = True
            ret = self._extract_feature(data_index, feature_index_list[0])
        else:
            self._extract_single_feature = False
            ret = tuple([self._extract_feature(data_index, j) for j in
                         feature_index_list])
        self.postprocess(item)
        return ret

    def check_type_feature_index(self, j):
        if j >= self.features_length():
            raise IndexError('index {} is out of bounds for axis 1 with '
                             'size {}'.format(j, self.features_length()))

    def _extract_feature(self, data_index, j):
        """Format `data_index` and call proper method to extract feature.

        Args:
            data_index (int, slice, list or numpy.ndarray):
            j (int or key):

        """
        self.check_type_feature_index(j)
        if isinstance(data_index, slice):
            try:
                return self.extract_feature_by_slice(data_index, j)
            except ExtractBySliceNotSupportedError:
                # Accessing by each index, copy occurs
                current, stop, step = data_index.indices(self.dataset_length)
                res = [self.extract_feature(i, j) for i in
                       six.moves.range(current, stop, step)]
        elif isinstance(data_index, (list, numpy.ndarray)):
            if isinstance(data_index[0], (bool, numpy.bool, numpy.bool_)):
                # Access by bool flag list
                if len(data_index) != self.dataset_length:
                    raise ValueError('Feature index wrong length {} instead of'
                                     ' {}'.format(len(data_index),
                                                  self.dataset_length))
                data_index = numpy.argwhere(data_index).ravel()

            if len(data_index) == 1:
                return self.extract_feature(data_index[0], j)
            else:
                res = [self.extract_feature(i, j) for i in data_index]
        else:
            return self.extract_feature(data_index, j)
        try:
            feature = numpy.asarray(res)
        except ValueError:
            feature = numpy.empty(len(res), dtype=object)
            feature[:] = res[:]
        return feature
