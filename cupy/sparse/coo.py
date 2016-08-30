import numpy
import scipy.sparse

import cupy
from cupy.cuda import cusparse


class coo_matrix(object):

    format = 'coo'

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

    def copy(self):
        return coo_matrix(
            (self.data.copy(), (self.row.copy(), self.col.copy())),
            shape=self.shape)

    def toarray(self, order=None, out=None):
        return self.tocsr().toarray()

    def tocoo(self, copy=False):
        return self

    def tocsc(self, copy=False):
        x = self.copy()
        x._sort_by_row()
        n = x.shape[1]
        indptr = cupy.empty(n + 1, 'i')
        cusparse.xcoo2csr(
            x._handle, x.col.data.ptr, x.nnz, n,
            indptr.data.ptr, cusparse.CUSPARSE_INDEX_BASE_ZERO)
        return cupy.sparse.csr.csc_matrix(
            (x.data, x.row, indptr), shape=x.shape)

    def tocsr(self, copy=False):
        x = self.copy()
        x._sort_by_row()
        m = x.shape[0]
        indptr = cupy.empty(m + 1, 'i')
        cusparse.xcoo2csr(
            x._handle, x.row.data.ptr, x.nnz, m,
            indptr.data.ptr, cusparse.CUSPARSE_INDEX_BASE_ZERO)
        return cupy.sparse.csr.csr_matrix(
            (x.data, x.col, indptr), shape=x.shape)

    def _sort_by_row(self):
        # Note that cuSPARSE's coo matrix is assumed to be sorted by row
        m, n = self.shape
        nnz = self.nnz

        buffer_size = cusparse.xcoosort_bufferSizeExt(
            self._handle, m, n, nnz, self.row.data.ptr, self.col.data.ptr)
        buf = cupy.empty(buffer_size, 'b')
        P = cupy.empty(nnz, 'i')
        cusparse.createIdentityPermutation(self._handle, nnz, P.data.ptr)
        cusparse.xcoosortByRow(
            self._handle, m, n, nnz, self.row.data.ptr, self.col.data.ptr,
            P.data.ptr, buf.data.ptr)
        cusparse.sgthr(
            self._handle, nnz, self.data.data.ptr, self.data.data.ptr,
            P.data.ptr, cusparse.CUSPARSE_INDEX_BASE_ZERO)

    def transpose(self, axes=None, copy=False):
        shape = self.shape[1], self.shape[0]
        return coo_matrix((self.data, (self.col, self.row)), shape=shape)
