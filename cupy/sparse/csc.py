import numpy
import scipy.sparse

import cupy
from cupy.cuda import cusparse


class csc_matrix(object):

    format = 'csc'

    def __init__(self, arg1, shape=None, dtype=None, copy=False):
        if len(arg1) == 3:
            data, indices, indptr = arg1
            assert len(data) == len(indices)
            self.data = data
            self.indices = indices
            self.indptr = indptr

            if shape is None:
                shape = (int(indices.max()) + 1, len(indptr) - 1)
        else:
            raise NotImplemented

        assert shape[1] == len(indptr) - 1

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
        return scipy.sparse.csc_matrix(
            (data, indices, indptr), shape=self.shape)

    def __str__(self):
        return str(self.get())

    @property
    def ndim(self):
        return 2

    @property
    def nnz(self):
        return len(self.data)

    def toarray(self, order=None, out=None):
        A = cupy.zeros(self.shape, 'f')
        cusparse.scsr2dense(
            self.handle, self.shape[1], self.shape[0], self._descr,
            self.data.data.ptr, self.indptr.data.ptr, self.indices.data.ptr,
            A.data.ptr, self.shape[1])
        return A

    def tocsc(self, copy=False):
        return self

    def tocsr(self, copy=False):
        m, n = self.shape
        nnz = self.nnz
        data = cupy.empty(nnz, 'f')
        indptr = cupy.empty(m + 1, 'i')
        indices = cupy.empty(nnz, 'i')
        cusparse.scsr2csc(
            self.handle, n, m, nnz, self.data.data.ptr,
            self.indptr.data.ptr, self.indices.data.ptr,
            data.data.ptr, indices.data.ptr, indptr.data.ptr,
            cusparse.CUSPARSE_ACTION_NUMERIC,
            cusparse.CUSPARSE_INDEX_BASE_ZERO)
        return cupy.sparse.csr_matrix((data, indices, indptr), shape=(m, n))

    def sort_indices(self):
        m, n = self.shape
        nnz = self.nnz

        buffer_size = cusparse.xcscsort_bufferSizeExt(
            self.handle, m, n, nnz, self.indptr.data.ptr, self.indices.data.ptr)
        buf = cupy.empty(buffer_size, 'b')
        P = cupy.empty(nnz, 'i')
        cusparse.createIdentityPermutation(self.handle, nnz, P.data.ptr)
        cusparse.xcscsort(
            self.handle, m, n, nnz, self._descr, self.indptr.data.ptr,
            self.indices.data.ptr, P.data.ptr, buf.data.ptr)
        cusparse.sgthr(
            self.handle, nnz, self.data.data.ptr, self.data.data.ptr,
            P.data.ptr, cusparse.CUSPARSE_INDEX_BASE_ZERO)

    def transpose(self, axes=None, copy=False):
        from cupy.sparse import csr

        shape = self.shape[1], self.shape[0]
        return csr.csr_matrix(
            (self.data, self.indices, self.indptr), shape=shape)
