import numpy
import six


class ExtractBySliceNotSupportedError(Exception):
    pass


class BaseIndexer(object):
    """Base class for Indexer
    """
    def __init__(self, *args, **kwargs):
        pass

    def __getitem__(self, item):
        raise NotImplementedError


class BaseFeaturesIndexer(BaseIndexer):
    """Base class for FeaturesIndexer
    
    Let `features` be the instance of `BaseFeaturesIndexer`, then 
    `features[i, j]` returns `i`-th dataset of `j`-th feature.
    
    `features[ind]` works same with `features[ind, :]`
    
    Note that the returned value will be numpy array, even though the 
    dataset is initilized with otherformat (e.g. list)
    
    """
    def __init__(self, dataset, *args, **kwargs):
        super(BaseFeaturesIndexer, self).__init__(*args, **kwargs)
        self.dataset = dataset

    @property
    def features_length(self):
        raise NotImplementedError

    @property
    def dataset_length(self):
        return len(self.dataset)

    @property
    def shape(self):
        return self.dataset_length, self.features_length

    def extract_feature_by_slice(self, slice_index, j):
        """Extracts `i`-th data's `j`-th feature,
        where `i` is in the range specified by `slice_index`

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

#    def extract_example(self, index, features_list):
#        """Extracts `index`-th example of `features_list`
#
#        Args:
#            index (int or slice or list or numpy.ndarray):
#            example should be extracted according to this index
#            features_list (list):
#
#        Returns: extracted features. They will be numpy array, even though the
#        dataset is initialized with other format (e.g. list)
#
#        """
#        # row_index must be one dimensional index
#        if isinstance(index, slice):
#            try:
#                # Accessing by slice, which is much efficient
#                res = [numpy.asarray(data[index]) for data in features_list]
#            except:
#                print('[DEBUG] slice index access failed...')
#                # Accessing by each index, copy occurs
#                current, stop, step = index.indices(len(features_list[0]))
#                res = [numpy.asarray([data[i] for i in
#                                      six.moves.range(current, stop, step)])
#                       for data in features_list]
#        elif isinstance(index, (list, numpy.ndarray)):
#            print('[DEBUG] type', type(index[0]))
#            if isinstance(index[0], (bool, numpy.bool, numpy.bool_)):
#                # Access by bool flag list
#                assert len(index) == len(features_list[0])
#                res = [numpy.asarray(data[index])
#                       for data in features_list]
#            else:
#                if len(index) == 1:
#                    res = [data[index] for data in features_list]
#                else:
#                    res = [numpy.concatenate([data[i:i+1] for i in index], axis=0)
#                           for data in features_list]
#                    #res = numpy.asarray([[data[i] for i in index]
#                    #                     for data in features_list])
#        else:
#            res = [numpy.asarray(data[index]) for data in features_list]
#        if len(res) == 1:
#            return res[0]
#        else:
#            return res
#
    def __getitem__(self, item):
        if isinstance(item, tuple):
            index_dim = len(item)
            # multi dimensional access
            if index_dim == 1:
                # currently, this is not expected...
                print('[WARNING], unexpected case...')
                data_index = item[0]
                feature_index_list = numpy.arange(self.features_length)
                #features_list = [self.extract_feature(j) for j in
                #                 range(self.features_length)]
                #return self.extract_example(item, features_list)
            elif index_dim == 2:
                # accessed by data.ix[:, 1]

                data_index, feature_index = item
                #row_index = item[0]
                #feature_index = item[1]
                if isinstance(feature_index, slice):
                    feature_index_list = numpy.arange(
                        *feature_index.indices(self.features_length)
                    )

                    #current, stop, step = feature_index.indices(self.features_length)
                    #col_datasets = [self.extract_feature(col) for col in
                    #                six.moves.range(current, stop, step)]
                elif isinstance(feature_index, (list, numpy.ndarray)):
                    if isinstance(feature_index[0],
                                  (bool, numpy.bool, numpy.bool_)):
                        assert len(feature_index) == self.features_length
                        feature_index_list = numpy.argwhere(feature_index
                                                            ).ravel()

                        #col_datasets = []
                        #for col, flag in enumerate(feature_index):
                        #    if flag:
                        #        col_datasets.append(self.extract_feature(col))
                    else:
                        feature_index_list = feature_index
                        #col_datasets = [self.extract_feature(col) for col
                        #                in feature_index]
                else:
                    # assuming int type
                    feature_index_list = [feature_index]
                    #col_datasets = [self.extract_feature(feature_index)]
                #return self.extract_example(row_index, col_datasets)
            else:
                raise IndexError('too many indices for features')
                #print('[Error] out of range, invalid index dimension')
        else:
            data_index = item
            feature_index_list = numpy.arange(self.features_length)
            #features_list = [self.extract_feature(j) for j in
            #                 range(self.features_length)]
            #return self.extract_example(item, features_list)
        if len(feature_index_list) == 1:
            return self._extract_feature(data_index, feature_index_list[0])
        else:
            return (self._extract_feature(data_index, j) for j in
                    feature_index_list)

    def _extract_feature(self, data_index, j):
        """Format `data_index` and call proper method to extract feature.
        
        Args:
            data_index (int, slice, list or numpy.ndarray):
            j (int):

        """
        if isinstance(data_index, slice):
            try:
                return self.extract_feature_by_slice(data_index, j)
            except ExtractBySliceNotSupportedError:
                print('[DEBUG] slice index access failed...')
                # Accessing by each index, copy occurs
                current, stop, step = data_index.indices(self.dataset_length)
                res = [self.extract_feature(i, j) for i in
                       six.moves.range(current, stop, step)]
                #res = [numpy.asarray([data[i] for i in
                #                      six.moves.range(current, stop, step)])
                #       for data in features_list]
        elif isinstance(data_index, (list, numpy.ndarray)):
            if isinstance(data_index[0], (bool, numpy.bool, numpy.bool_)):
                # Access by bool flag list
                assert len(data_index) == self.dataset_length
                data_index = numpy.argwhere(data_index).ravel()

            if len(data_index) == 1:
                return self.extract_feature(data_index[0], j)
            else:
                res = [self.extract_feature(i, j) for i in data_index]
                #res = numpy.asarray([[data[i] for i in index]
                #                     for data in features_list])
        else:
            return self.extract_feature(data_index, j)
        return numpy.asarray(res)
