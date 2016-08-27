import numpy
import scipy.sparse

import cupy
from cupy.cuda import cusparse


class coo_matrix(object):

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        if len(arg1) == 2:
            data, (row, col) = arg1
            assert len(data) == len(row) == len(col)

            self.data = data
            self.row = row
            self.col = col

            if shape is None:
                shape = (row.max() + 1, col.max() + 1)

        self._shape = shape
        self._handle = cusparse.create()

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return 2

    @property
    def nnz(self):
        return len(self.data)

    def get(self, stream=None):
        data = self.data.get(stream)
        row = self.row.get(stream)
        col = self.col.get(stream)
        return scipy.sparse.coo_matrix(
            (data, (row, col)), shape=self.shape)

    def __str__(self):
        return str(self.get())

    def toarray(self, order=None, out=None):
        return self.tocsr().toarray()

    def tocsr(self, copy=False):
        m = self.shape[0]
        indptr = cupy.empty(m + 1, 'i')
        cusparse.xcoo2csr(
            self._handle, self.row.data.ptr, self.nnz, m,
            indptr.data.ptr, cusparse.CUSPARSE_INDEX_BASE_ZERO)
        return cupy.sparse.csr.csr_matrix(
            (self.data, self.col, indptr), shape=self.shape)
