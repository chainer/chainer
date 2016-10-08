import numpy
import scipy.sparse

import cupy
from cupy.cuda import cusparse
from cupy.sparse import base


class csr_matrix(base.spmatrix):

    format = 'csr'

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
        self._shape = shape

    def get(self, stream=None):
        data = self.data.get(stream)
        indices = self.indices.get(stream)
        indptr = self.indptr.get(stream)
        return scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=self.shape)

    def getnnz(self, axis=None):
        if axis is None:
            return len(self.data)
        else:
            raise NotImplemented

    def __getitem__(self, key):
        i, j = key
        start = self.indptr[i]
        end = self.indptr[i + 1]
        answer = cupy.zeros((), 'f')
        kern = cupy.ElementwiseKernel(
            'T d, S ind, int32 col', 'raw T answer',
            'if (ind == col) atomicAdd(&answer[0], d);',
            'csr_getitem')
        kern(self.data[start:end], self.indices[start:end], j, answer)
        return answer[()]

    def tocoo(self, copy=False):
        m = self.shape[0]
        nnz = self.nnz
        row = cupy.empty(nnz, 'i')
        cusparse.xcsr2coo(
            self.handle, self.indptr.data.ptr, nnz, m, row.data.ptr,
            cusparse.CUSPARSE_INDEX_BASE_ZERO)
        return cupy.sparse.coo.coo_matrix(
            (self.data, (row, self.indices)), shape=self.shape)

    def tocsr(self, copy=False):
        return self

    def tocsc(self, copy=False):
        m, n = self.shape
        nnz = self.nnz
        data = cupy.empty(nnz, 'f')
        indptr = cupy.empty(n + 1, 'i')
        indices = cupy.empty(nnz, 'i')
        cusparse.scsr2csc(
            self.handle, m, n, nnz, self.data.data.ptr,
            self.indptr.data.ptr, self.indices.data.ptr,
            data.data.ptr, indices.data.ptr, indptr.data.ptr,
            cusparse.CUSPARSE_ACTION_NUMERIC,
            cusparse.CUSPARSE_INDEX_BASE_ZERO)
        return cupy.sparse.csc_matrix((data, indices, indptr), shape=(m, n))

    def toarray(self, order=None, out=None):
        A = cupy.zeros((self.shape[1], self.shape[0]), 'f')
        cusparse.scsr2dense(
            self.handle, self.shape[0], self.shape[1], self._descr,
            self.data.data.ptr, self.indptr.data.ptr, self.indices.data.ptr,
            A.data.ptr, self.shape[0])
        return A.T

    def dot(self, x):
        y = cupy.zeros((self.shape[0]), 'f')
        alpha = numpy.array(1, 'f').ctypes
        beta = numpy.array(0, 'f').ctypes
        cusparse.scsrmv(
            self.handle, cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
            self.shape[0], self.shape[1], self.nnz, alpha.data, self._descr,
            self.data.data.ptr, self.indptr.data.ptr, self.indices.data.ptr,
            x.data.ptr, beta.data, y.data.ptr)

        return y

    def __add__(self, x):
        assert self.shape == x.shape

        m, n = self.shape
        nnz = numpy.empty((), 'i')
        cusparse.setPointerMode(
            self.handle, cusparse.CUSPARSE_POINTER_MODE_HOST)

        c_descr = cusparse.createMatDescr()
        c_indptr = cupy.empty(m + 1, 'i')

        cusparse.xcsrgeamNnz(
            self.handle, m, n,
            self._descr, self.nnz, self.indptr.data.ptr, self.indices.data.ptr,
            x._descr, x.nnz, x.indptr.data.ptr, x.indices.data.ptr,
            c_descr, c_indptr.data.ptr, nnz.ctypes.data)

        c_indices = cupy.empty(int(nnz), 'i')
        c_data = cupy.empty(int(nnz), 'f')
        one = numpy.array(1, 'f').ctypes
        cusparse.scsrgeam(
            self.handle, m, n, one.data,
            self._descr, self.nnz, self.data.data.ptr,
            self.indptr.data.ptr, self.indices.data.ptr,
            one.data, x._descr, x.nnz, x.data.data.ptr, x.indptr.data.ptr,
            x.indices.data.ptr, c_descr, c_data.data.ptr, c_indptr.data.ptr,
            c_indices.data.ptr)

        return csr_matrix((c_data, c_indices, c_indptr), shape=self.shape)

    def __mul__(self, x):
        assert self.shape[1] == x.shape[0]
        m, n = self.shape
        k = x.shape[1]

        transA = cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        transB = cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE

        nnz = numpy.empty((), 'i')
        cusparse.setPointerMode(
            self.handle, cusparse.CUSPARSE_POINTER_MODE_HOST)

        c_descr = cusparse.createMatDescr()
        c_indptr = cupy.empty(m + 1, 'i')

        cusparse.xcsrgemmNnz(
            self.handle, transA, transB, m, n, k, self._descr, self.nnz,
            self.indptr.data.ptr, self.indices.data.ptr, x._descr, x.nnz,
            x.indptr.data.ptr, x.indices.data.ptr, c_descr,
            c_indptr.data.ptr, nnz.ctypes.data)

        c_indices = cupy.empty(int(nnz), 'i')
        c_data = cupy.empty(int(nnz), 'f')
        cusparse.scsrgemm(
            self.handle, transA, transB, m, n, k, self._descr, self.nnz,
            self.data.data.ptr, self.indptr.data.ptr, self.indices.data.ptr,
            x._descr, x.nnz, x.data.data.ptr, x.indptr.data.ptr,
            x.indices.data.ptr, c_descr, c_data.data.ptr, c_indptr.data.ptr,
            c_indices.data.ptr)

        return csr_matrix((c_data, c_indices, c_indptr), shape=(m, k))

    def sort_indices(self):
        m, n = self.shape
        nnz = self.nnz

        buffer_size = cusparse.xcsrsort_bufferSizeExt(
            self.handle, m, n, nnz, self.indptr.data.ptr, self.indices.data.ptr)
        buf = cupy.empty(buffer_size, 'b')
        P = cupy.empty(nnz, 'i')
        cusparse.createIdentityPermutation(self.handle, nnz, P.data.ptr)
        cusparse.xcsrsort(
            self.handle, m, n, nnz, self._descr, self.indptr.data.ptr,
            self.indices.data.ptr, P.data.ptr, buf.data.ptr)
        cusparse.sgthr(
            self.handle, nnz, self.data.data.ptr, self.data.data.ptr,
            P.data.ptr, cusparse.CUSPARSE_INDEX_BASE_ZERO)

    def transpose(self, axes=None, copy=False):
        from cupy.sparse import csc

        shape = self.shape[1], self.shape[0]
        return csc.csc_matrix(
            (self.data, self.indices, self.indptr), shape=shape)
