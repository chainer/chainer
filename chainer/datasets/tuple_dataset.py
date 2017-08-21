import numpy
import six

from chainer.dataset.indexer import BaseIxIndexer


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
        self._ix = TupleDatasetIxIndexer(self._datasets)

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
    def ix(self):
        return self._ix


class TupleDatasetIxIndexer(BaseIxIndexer):
    """
    
    `dataset.ix[ind]` works same with `dataset.ix[ind, :]`
    
    """

    def __init__(self, datasets):
        super(TupleDatasetIxIndexer, self).__init__()
        self.datasets = datasets

    def extract_row(self, row_index, col_datasets):
        # row_index must be one dimensional index

        if isinstance(row_index, slice):
            # Accessing by slice, which is much efficient
            res = [numpy.asarray(data[row_index]) for data in col_datasets]

            # Accessing by each index, copy occurs
            #current, stop, step = row_index.indices(len(col_datasets[0]))
            #res = [numpy.asarray([data[i] for i in six.moves.range(current, stop, step)])
            #        for data in col_datasets]
        elif isinstance(row_index, list) or isinstance(row_index, numpy.ndarray):
            print('type', type(row_index[0]))
            if isinstance(row_index[0], (bool, numpy.bool, numpy.bool_)):
                # Access by bool flag list
                assert len(row_index) == len(col_datasets[0])
                res = [numpy.asarray(data[row_index])
                       for data in col_datasets]
            else:
                res = [numpy.concatenate([data[i] for i in row_index], axis=0)
                       for data in col_datasets]
        else:
            res = [numpy.asarray(data[row_index]) for data in col_datasets]
        if len(res) == 1:
            return res[0]
        else:
            return res

    def __getitem__(self, item):
        if isinstance(item, tuple):
            index_dim = len(item)
            # multi dimensional access
            if index_dim == 1:
                # currently, this is not expected...
                print('WARNING, this case not implemented yet...')
            elif index_dim == 2:
                # accessed by data.ix[:, 1]
                row_index = item[0]
                col_index = item[1]
                if isinstance(col_index, slice):
                    current, stop, step = col_index.indices(len(self.datasets))
                    col_datasets = [self.datasets[col] for col in
                                    six.moves.range(current, stop, step)]
                elif isinstance(col_index, list) or isinstance(col_index, numpy.ndarray):
                    if isinstance(col_index[0],
                                  (bool, numpy.bool, numpy.bool_)):
                        assert len(col_index) == len(self.datasets)
                        col_datasets = []
                        for col, flag in enumerate(col_index):
                            if flag:
                                col_datasets.append(self.datasets[col])
                    else:
                        col_datasets = [self.datasets[col] for col in col_index]
                else:
                    col_datasets = [self.datasets[col_index]]
                return self.extract_row(row_index, col_datasets)
            else:
                print('[Error] out of range, invalid index dimension')
        else:
            # Accessing all feature in dataset with specified id
            """
            1. int
            2. slice
            3. list, ndarray (advanced indexing) 
            4. boolean list
            """
            return self.extract_row(item, self.datasets)



