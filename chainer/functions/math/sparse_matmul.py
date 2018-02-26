import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check
import numpy
import warnings

try:
    from scipy import sparse
except ImportError:
    warnings.warn("SciPy seems not available on your system. A CPU"
                  " implementation of sparse_matmul uses SciPy, so that you"
                  " cannot use sparse_matmul on CPU.")


class sparse_coo_matrix(object):

    def __init__(self, data, row, col, shape, use_variable=False):
        self.data = data
        if use_variable:
            self.data = chainer.Variable(self.data)
        self.row = row
        self.col = col
        self.shape = shape  # (row, col)

    def to_dense(self):
        xp = cuda.get_array_module(self.data)
        if self.data.ndim == 1:
            x = xp.zeros(self.shape, dtype=self.data.dtype)
            x[self.row, self.col] = self.data
            return x
        if self.data.ndim == 2:
            nb = self.data.shape[0]
            x = xp.zeros((nb, self.shape[0], self.shape[1]),
                         dtype=self.data.dtype)
            for i in range(nb):
                nnz = len(xp.where(self.data[i] != 0)[0])
                x[i, self.row[i, :nnz], self.col[i, :nnz]] = self.data[i,
                                                                       0:nnz]
            return x


def sparse_dense2coo(x, ldnz=None, use_variable=False):
    xp = cuda.get_array_module(x)
    if x.ndim == 2:
        _row, _col = xp.where(x != 0)
        nnz = len(_row)
        if ldnz is None or ldnz < nnz:
            ldnz = nnz
        data = xp.zeros((ldnz), dtype=x.dtype)
        row = xp.full((ldnz), -1, dtype=xp.int32)
        col = xp.full((ldnz), -1, dtype=xp.int32)
        data[0:nnz] = x[_row, _col]
        row[0:nnz] = xp.array(_row).astype(xp.int32)
        col[0:nnz] = xp.array(_col).astype(xp.int32)
        shape = x.shape
        return sparse_coo_matrix(data, row, col, shape, use_variable)
    elif x.ndim == 3:
        # first axis is batch axis
        nb = x.shape[0]
        if ldnz is None:
            ldnz = 0
        for i in range(nb):
            ldnz = max(ldnz, len(xp.where(x[i] != 0)[0]))
        data = xp.empty((nb, ldnz), dtype=x.dtype)
        row = xp.empty((nb, ldnz), dtype=xp.int32)
        col = xp.empty((nb, ldnz), dtype=xp.int32)
        for i in range(nb):
            coo = sparse_dense2coo(x[i], ldnz)
            data[i] = coo.data
            row[i] = coo.row
            col[i] = coo.col
        shape = x.shape[1:]
        return sparse_coo_matrix(data, row, col, shape, use_variable)
    else:
        raise ValueError('ndim of x must be 2 or 3.')


def _sparse_matmul(sp_data, sp_row, sp_col, sp_shape, dn,
                   transa, transb, transc, dtype=None):
    if dtype is None:
        dtype = numpy.result_type(sp_data.dtype, dn.dtype)

    A_data = sp_data
    if transa:
        A_row = sp_col
        A_col = sp_row
        A_shape = [sp_shape[1], sp_shape[0]]
    else:
        A_row = sp_row
        A_col = sp_col
        A_shape = sp_shape
    if transb:
        B = dn.swapaxes(-1, -2)
    else:
        B = dn

    xp = cuda.get_array_module(B)
    if xp is numpy:
        C = _sparse_matmul_cpu(A_data, A_row, A_col, A_shape, B, dtype)
    else:
        C = _sparse_matmul_gpu(A_data, A_row, A_col, A_shape, B, dtype)

    if transc:
        C = C.swapaxes(-1, -2)
    return C


def _sparse_matmul_cpu(A_data, A_row, A_col, A_shape, B, dtype):
    # A_shape: (_m, _k)
    # B.shape: ((nb,) _k, _n)
    # A_data/row/col.shape: ((nb,) ldnz)
    _m, _k = A_shape
    _n = B.shape[-1]
    if B.ndim == 2:
        sp_A = sparse.coo_matrix((A_data, (A_row, A_col)), shape=(_m, _k))
        C = sp_A.dot(B).astype(dtype, copy=False)
    else:
        nb = B.shape[0]
        C = numpy.empty((nb, _m, _n), dtype=dtype)
        for i in range(nb):
            nnz = len(numpy.where(A_row[i] >= 0)[0])
            sp_A = sparse.coo_matrix((A_data[i, :nnz],
                                      (A_row[i, :nnz], A_col[i, :nnz])),
                                     shape=(_m, _k))
            C[i] = sp_A.dot(B[i]).astype(dtype, copy=False)

    return C


def _sparse_matmul_gpu(A_data, A_row, A_col, A_shape, B, dtype):
    cupy_dtype = dtype
    if cupy_dtype == numpy.float16:
        cupy_dtype = numpy.float32
        # fp32 is used in cupy kernel because fp16 atomicAdd is not supported

    # A_shape: (_m, _k)
    # B.shape: ((nb,) _k, _n)
    # A_data/row/col.shape: ((nb,) ldnz)
    _m, _k = A_shape
    _n = B.shape[-1]
    ldnz = A_data.shape[-1]
    if B.ndim == 2:
        nb = 1
        C = cuda.cupy.zeros((_m, _n), dtype=cupy_dtype)
    else:
        nb = B.shape[0]
        C = cuda.cupy.zeros((nb, _m, _n), dtype=cupy_dtype)

    nthreads = nb * ldnz * _n
    _cupy_sparse_matmul()(nb, _m, _n, _k, ldnz, A_data, A_row, A_col,
                          B, C, size=nthreads)

    return C.astype(dtype, copy=False)


def _cupy_sparse_matmul():
    return cuda.cupy.ElementwiseKernel(
        'int32 nb, int32 _m, int32 _n, int32 _k, int32 nnz, \
         raw A A_data, raw T A_row, raw T A_col, \
         raw B _B, raw C _C',
        '',
        '''
        int i_n = (i % _n);
        int i_A = (i / _n);
        int i_b = (i / _n) / nnz;
        if (i_b >= nb) {
            continue;
        }
        int i_k = A_col[i_A];
        if (i_k < 0) {
            continue;
        }
        assert(i_k < _k);
        int i_m = A_row[i_A];
        if (i_m < 0) {
            continue;
        }
        assert(i_m < _m);
        int i_B = i_n + _n * (i_k + _k * i_b);
        int i_C = i_n + _n * (i_m + _m * i_b);
        A val_A = A_data[i_A];
        atomicAdd(&_C[i_C], (C)(_B[i_B] * val_A));
        ''',
        'sparse_matmul')


class SparseMatMul(function_node.FunctionNode):

    def __init__(self, sp_row, sp_col, sp_shape,
                 transa=False, transb=False, transc=False,
                 dtype=None):
        if sp_row.ndim != sp_col.ndim:
            raise ValueError('ndim of sp_row and sp_col must be the same.')
        if sp_row.ndim != 1 and sp_row.ndim != 2:
            raise ValueError('ndim of sp_row and sp_col must be one or two.')
        for i in range(sp_row.ndim):
            if sp_row.shape[i] != sp_col.shape[i]:
                _msg = 'shape of sp_row and sp_col must be the same.'
                raise ValueError(_msg)
        if len(sp_shape) != 2:
            raise ValueError('len(sp_shape) must be two.')
        self.sp_row = sp_row  # ((nb,) ldnz)
        self.sp_col = sp_col  # ((nb,) ldnz)
        self.sp_shape = sp_shape  # (_m, _k) when transa is False
        self.transa = transa
        self.transb = transb
        self.transc = transc
        self.dtype = dtype

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        sp_type, dn_type = in_types
        # sp_type.shape: ((nb,) ldnz)
        # dn_type.shape: ((nb,) _k, _n) when transb is False
        type_check.expect(
            sp_type.dtype.kind == 'f',
            dn_type.dtype.kind == 'f',
            dn_type.ndim >= 2,
            dn_type.ndim <= 3,
            sp_type.ndim == dn_type.ndim - 1,
            sp_type.shape[-1] == self.sp_row.shape[-1],
        )
        if self.transa is False and self.transb is False:
            type_check.expect(
                self.sp_shape[-1] == dn_type.shape[-2],
            )
        if self.transa is False and self.transb is True:
            type_check.expect(
                self.sp_shape[-1] == dn_type.shape[-1],
            )
        if self.transa is True and self.transb is False:
            type_check.expect(
                self.sp_shape[-2] == dn_type.shape[-2],
            )
        if self.transa is True and self.transb is True:
            type_check.expect(
                self.sp_shape[-2] == dn_type.shape[-1],
            )
        dn_ndim = type_check.eval(dn_type.ndim)
        if dn_ndim == 3:
            type_check.expect(
                sp_type.shape[0] == self.sp_row.shape[0],
                dn_type.shape[0] == self.sp_row.shape[0],
            )

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        sp, dn = inputs
        c = _sparse_matmul(sp, self.sp_row, self.sp_col, self.sp_shape, dn,
                           self.transa, self.transb, self.transc, self.dtype)
        return utils.force_array(c, self.dtype),

    def backward(self, indexes, grad_outputs):
        sp, dn = self.get_retained_inputs()
        g_c, = grad_outputs

        g_sp = None
        if 0 in indexes:
            g_sp = SparseMatMulGradSP(self.sp_row, self.sp_col, self.sp_shape,
                                      self.transc, not self.transb,
                                      self.transa,
                                      dtype=sp.dtype).apply((g_c, dn))[0]

        g_dn = None
        if 1 in indexes:
            g_dn = SparseMatMul(self.sp_row, self.sp_col, self.sp_shape,
                                not self.transa, self.transc, self.transb,
                                dtype=dn.dtype).apply((sp, g_c))[0]

        return g_sp, g_dn


def _sparse_matmul_gradsp(a, b, c_row, c_col, c_shape,
                          transa, transb, transc, dtype):
    if dtype is None:
        dtype = numpy.result_type(a.dtype, b.dtype)

    if transa:
        A = a.swapaxes(-1, -2)
    else:
        A = a
    if transb:
        B = b.swapaxes(-1, -2)
    else:
        B = b
    if transc:
        C_row = c_col
        C_col = c_row
    else:
        C_row = c_row
        C_col = c_col

    xp = cuda.get_array_module(A)
    if xp is numpy:
        return _sparse_matmul_gradsp_cpu(A, B, C_row, C_col, dtype)
    else:
        return _sparse_matmul_gradsp_gpu(A, B, C_row, C_col, dtype)


def _sparse_matmul_gradsp_cpu(A, B, C_row, C_col, dtype):
    # A.shape: ((nb,) _m, _k)
    # B.shape: ((nb,) _k, _n)
    # C_row/col.shape: ((nb,) ldnz)
    _m, _k = A.shape[-2:]
    ldnz = C_row.shape[-1]
    if A.ndim == 2:
        C_data = numpy.zeros((ldnz), dtype=dtype)
        C = numpy.matmul(A, B).astype(dtype, copy=False)
        nnz = len(numpy.where(C_row >= 0)[0])
        C_data[:nnz] = C[C_row[:nnz], C_col[:nnz]]
    else:
        nb = A.shape[0]
        C_data = numpy.zeros((nb, ldnz), dtype=dtype)
        C = numpy.matmul(A, B).astype(dtype, copy=False)
        for i in range(nb):
            nnz = len(numpy.where(C_row[i] >= 0)[0])
            C_data[i, :nnz] = C[i, C_row[i, :nnz], C_col[i, :nnz]]

    return C_data


def _sparse_matmul_gradsp_gpu(A, B, C_row, C_col, dtype):
    # A.shape: ((nb,) _m, _k)
    # B.shape: ((nb,) _k, _n)
    # C_row/col.shape: ((nb,) ldnz)
    _m, _k = A.shape[-2:]
    _n = B.shape[-1]
    ldnz = C_row.shape[-1]
    if A.ndim == 2:
        nb = 1
        C_data = cuda.cupy.zeros((ldnz), dtype=dtype)
    else:
        nb = A.shape[0]
        C_data = cuda.cupy.zeros((nb, ldnz), dtype=dtype)

    nthreads = nb * ldnz
    _cupy_sparse_matmul_gradsp()(nb, _m, _n, _k, ldnz, A, B,
                                 C_data, C_row, C_col, size=nthreads)

    return C_data


def _cupy_sparse_matmul_gradsp():
    return cuda.cupy.ElementwiseKernel(
        'int32 nb, int32 _m, int32 _n, int32 _k, int32 nnz, \
         raw A _A, raw B _B, \
         raw C C_data, raw T C_row, raw T C_col',
        '',
        '''
        int i_nz = (i % nnz);
        int i_b = (i / nnz);
        if (i_b >= nb) {
            continue;
        }
        int i_C = i;
        int i_m = C_row[i_C];
        if (i_m < 0) {
            continue;
        }
        assert(i_m < _m);
        int i_n = C_col[i_C];
        if (i_n < 0) {
            continue;
        }
        assert(i_n < _n);
        C val_C = 0.0;
        for (int i_k = 0; i_k < _k; i_k++) {
            int i_A = i_k + _k * (i_m + _m * i_b);
            int i_B = i_n + _n * (i_k + _k * i_b);
            A val_A = _A[i_A];
            B val_B = _B[i_B];
            val_C += (C)(val_A * val_B);
        }
        C_data[i_C] = val_C;
        ''',
        'sparse_matmul_gradsp')


class SparseMatMulGradSP(function_node.FunctionNode):

    def __init__(self, sp_row, sp_col, sp_shape,
                 transa=False, transb=False, transc=False,
                 dtype=None):
        if sp_row.ndim != sp_col.ndim:
            raise ValueError('ndim of sp_row and sp_col must be the same.')
        if sp_row.ndim != 1 and sp_row.ndim != 2:
            raise ValueError('ndim of sp_row and sp_col must be one or two.')
        for i in range(sp_row.ndim):
            if sp_row.shape[i] != sp_col.shape[i]:
                _msg = 'shape of sp_row and sp_col must be the same.'
                raise ValueError(_msg)
        if len(sp_shape) != 2:
            raise ValueError('len(sp_shape) must be two.')
        self.sp_row = sp_row
        self.sp_col = sp_col
        self.sp_shape = sp_shape
        self.transa = transa
        self.transb = transb
        self.transc = transc
        self.dtype = dtype

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        a_type, b_type = in_types
        type_check.expect(
            a_type.dtype.kind == 'f',
            b_type.dtype.kind == 'f',
            a_type.ndim >= 2,
            a_type.ndim <= 3,
            a_type.ndim == b_type.ndim,
        )
        a_ndim = type_check.eval(a_type.ndim)
        if a_ndim == 3:
            type_check.expect(
                a_type.shape[0] == self.sp_row.shape[0],
                b_type.shape[0] == self.sp_row.shape[0],
            )

    def forward(self, inputs):
        self.retain_inputs((0, 1))
        a, b = inputs
        c = _sparse_matmul_gradsp(a, b,
                                  self.sp_row, self.sp_col, self.sp_shape,
                                  self.transa, self.transb, self.transc,
                                  self.dtype)
        return utils.force_array(c),

    def backward(self, indexes, grad_outputs):
        a, b = self.get_retained_inputs()
        g_sp, = grad_outputs

        g_a = None
        if 0 in indexes:
            g_a = SparseMatMul(self.sp_row, self.sp_col, self.sp_shape,
                               self.transc, not self.transb, self.transa,
                               dtype=a.dtype).apply((g_sp, b))[0]

        g_b = None
        if 0 in indexes:
            g_b = SparseMatMul(self.sp_row, self.sp_col, self.sp_shape,
                               not self.transc, self.transa, not self.transb,
                               dtype=b.dtype).apply((g_sp, a))[0]

        return g_a, g_b


def sparse_matmul(a, b, transa=False, transb=False):
    """Computes the batched multiplication of sparse and dense matrix.

    The following use cases are supported:

        1. C (dense) = A (sparse) * B (dense)
        2. C (dense) = A (dense) * B (sparse)

    Args:
        a (Variable or sparse_coo_matrix): The left operand of mat-mul.
        b (Variable or sparse_coo_matrix): The right operand of mat-mul.
        transa (bool): If ``True``, each matrix in ``a`` will be transposed.
        transb (bool): If ``True``, each matrix in ``b`` will be transposed.

    Returns:
        _chainer.Variable or sparse_coo_matrix: Result of mat-mul.
    """
    if (isinstance(a, sparse_coo_matrix) and
            isinstance(b, (chainer.Variable, numpy.ndarray, cuda.ndarray))):
        return SparseMatMul(a.row, a.col, a.shape,
                            transa=transa,
                            transb=transb,
                            transc=False).apply((a.data, b))[0]
    elif (isinstance(a, (chainer.Variable, numpy.ndarray, cuda.ndarray)) and
          isinstance(b, sparse_coo_matrix)):
        return SparseMatMul(b.row, b.col, b.shape,
                            transa=not transb,
                            transb=not transa,
                            transc=True).apply((b.data, a))[0]
    else:
        _msg = 'This combination of type of inputs is not supported.\n'
        _msg += '    a: {}\n'.format(type(a))
        _msg += '    b: {}\n'.format(type(b))
        raise ValueError(_msg)
