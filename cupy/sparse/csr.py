import numpy
import scipy.sparse

import cupy
from cupy.cuda import cusparse


class csr_matrix(object):

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        if len(arg1) == 3:
            data, indices, indptr = arg1
            assert len(data) == len(indices)
            self.data = data
            self.indices = indices
            self.indptr = indptr

            if shape is None:
                shape = (len(indptr) - 1, indices.max() + 1)
        else:
            raise NotImplemented

        assert shape[0] == len(indptr) - 1

        self.handle = cusparse.create()
        self._descr = cusparse.createMatDescr()
        cusparse.setMatType(self._descr, cusparse.CUSPARSE_MATRIX_TYPE_GENERAL)
        cusparse.setMatIndexBase(
            self._descr, cusparse.CUSPARSE_INDEX_BASE_ZERO)
        self.shape = shape

    def get(self, stream=None):
        data = self.data.get(stream)
        indices = self.indices.get(stream)
        indptr = self.indptr.get(stream)
        return scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=self.shape)

    def __str__(self):
        return str(self.get())

    @property
    def nnz(self):
        return len(self.data)
