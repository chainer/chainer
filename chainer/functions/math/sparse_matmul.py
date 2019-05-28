import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check

try:
    from scipy import sparse
    _scipy_available = True
except ImportError:
    _scipy_available = False


def _coo_matmul(sp_data, sp_row, sp_col, sp_shape, sp_order,
                dn, transa, transb, transc, dtype=None):
    if dtype is None:
        dtype = numpy.result_type(sp_data.dtype, dn.dtype)

    A_data = sp_data
    if transa:
        A_row = sp_col
        A_col = sp_row
        A_shape = (sp_shape[1], sp_shape[0])
        if sp_order == 'C':
            A_order = 'F'
        elif sp_order == 'F':
            A_order = 'C'
        else:
            A_order = sp_order
    else:
        A_row = sp_row
        A_col = sp_col
        A_shape = sp_shape
        A_order = sp_order
    if transb:
        B = dn.swapaxes(-1, -2)
    else:
        B = dn

    xp = backend.get_array_module(A_data, B)
    if xp is numpy:
        C = _coo_matmul_cpu(A_data, A_row, A_col, A_shape, B, dtype)
    else:
        C = _coo_matmul_gpu(A_data, A_row, A_col, A_shape, A_order,
                            B, dtype)

    if transc:
        C = C.swapaxes(-1, -2)
    return C


def _coo_matmul_cpu(A_data, A_row, A_col, A_shape, B, dtype):
    # A_shape: (_m, _k)
    # B.shape: ((nb,) _k, _n)
    # A_data/row/col.shape: ((nb,) ldnz)
    if not _scipy_available:
        msg = 'SciPy seems to be unavailable on your system. A CPU' \
              ' implementation of sparse_matmul uses SciPy, so you' \
              ' cannot use sparse_matmul on the CPU.'
        raise RuntimeError(msg)

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


def _coo_matmul_gpu(A_data, A_row, A_col, A_shape, A_order, B, dtype):
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

    if A_order == 'C':
        # A chunk is the number of non-zero elements handled by a single GPU
        # thread. If contiguous non-zero elemets are related to the same
        # location of the output matrix and they are processed in the same
        # thread, number of atomic-add operations can be reduced.
        chunk = max(ldnz // _m, 1)
    else:
        chunk = 1
    nthreads = (nb * ldnz + chunk - 1) // chunk * _n
    _cupy_coo_matmul()(nb, _m, _n, _k, ldnz, chunk,
                       A_data, A_row, A_col, B, C,
                       size=nthreads)

    return C.astype(dtype, copy=False)


def _cupy_coo_matmul():
    utils.nondeterministic('atomicAdd')
    return cuda.elementwise(
        'int32 nb, int32 _m, int32 _n, int32 _k, int32 nnz, int32 chunk, \
         raw A A_data, raw T A_row, raw T A_col, \
         raw B _B',
        'raw C _C',
        '''
        int i_n = (i % _n);
        int i0 = (i / _n) * chunk;
        int i_C = -1;
        C val_C = 0;
        for (int i1 = 0; i1 < chunk; i1++) {
            int i_A = i0 + i1;
            int i_b = i_A / nnz;
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
            int i_C_now = i_n + _n * (i_m + _m * i_b);
            A val_A = A_data[i_A];
            B val_B = _B[i_B];
            C val_C_now = static_cast<C>(val_A * val_B);
            if (i_C >= 0 && i_C != i_C_now) {
                atomicAdd(&_C[i_C], val_C);
                val_C = 0;
            }
            i_C = i_C_now;
            val_C += val_C_now;
        }
        if (i_C >= 0) {
            atomicAdd(&_C[i_C], val_C);
        }
        ''',
        'coo_matmul')


class CooMatMul(function_node.FunctionNode):

    def __init__(self, sp_row, sp_col, sp_shape, sp_order='other',
                 transa=False, transb=False, transc=False, dtype=None):
        if sp_row.ndim != sp_col.ndim:
            raise ValueError('ndim of sp_row and sp_col must be the same.')
        if sp_row.ndim != 1 and sp_row.ndim != 2:
            raise ValueError('ndim of sp_row and sp_col must be one or two.')
        for i in range(sp_row.ndim):
            if sp_row.shape[i] != sp_col.shape[i]:
                msg = 'shape of sp_row and sp_col must be the same.'
                raise ValueError(msg)
        if len(sp_shape) != 2:
            raise ValueError('len(sp_shape) must be two.')
        self.sp_row = sp_row  # ((nb,) ldnz)
        self.sp_col = sp_col  # ((nb,) ldnz)
        self.sp_shape = sp_shape  # (_m, _k) when transa is False
        self.sp_order = sp_order
        self.transa = transa
        self.transb = transb
        self.transc = transc
        self.dtype = dtype

    def check_type_forward(self, in_types):
        type_check._argname(in_types, ('sp', 'dn'))
        sp_type, dn_type = in_types
        # sp_type.shape: ((nb,) ldnz)
        # dn_type.shape: ((nb,) _k, _n) when transb is False
        sp_k_axis = -1
        if self.transa:
            sp_k_axis = -2
        dn_k_axis = -2
        if self.transb:
            dn_k_axis = -1
        type_check.expect(
            sp_type.dtype.kind == 'f',
            dn_type.dtype.kind == 'f',
            dn_type.ndim >= 2,
            dn_type.ndim <= 3,
            sp_type.ndim == dn_type.ndim - 1,
            sp_type.shape[-1] == self.sp_row.shape[-1],
            self.sp_shape[sp_k_axis] == dn_type.shape[dn_k_axis],
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
        c = _coo_matmul(sp, self.sp_row, self.sp_col, self.sp_shape,
                        self.sp_order, dn,
                        self.transa, self.transb, self.transc, self.dtype)
        return utils.force_array(c, self.dtype),

    def backward(self, indexes, grad_outputs):
        sp, dn = self.get_retained_inputs()
        g_c, = grad_outputs
        ret = []
        if 0 in indexes:
            g_sp = CooMatMulGradSP(self.sp_row, self.sp_col, self.sp_shape,
                                   self.sp_order,
                                   self.transc, not self.transb, self.transa,
                                   dtype=sp.dtype).apply((g_c, dn))[0]
            ret.append(g_sp)
        if 1 in indexes:
            g_dn = CooMatMul(self.sp_row, self.sp_col, self.sp_shape,
                             self.sp_order,
                             not self.transa, self.transc, self.transb,
                             dtype=dn.dtype).apply((sp, g_c))[0]
            ret.append(g_dn)
        return ret


def _coo_matmul_gradsp(a, b, c_row, c_col, c_shape, transa, transb, transc,
                       dtype):
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

    xp = backend.get_array_module(A, B)
    if xp is numpy:
        return _coo_matmul_gradsp_cpu(A, B, C_row, C_col, dtype)
    else:
        return _coo_matmul_gradsp_gpu(A, B, C_row, C_col, dtype)


def _coo_matmul_gradsp_cpu(A, B, C_row, C_col, dtype):
    # A.shape: ((nb,) _m, _k)
    # B.shape: ((nb,) _k, _n)
    # C_row/col.shape: ((nb,) ldnz)
    _m, _k = A.shape[-2:]
    ldnz = C_row.shape[-1]
    if hasattr(numpy, 'matmul'):
        C = numpy.matmul(A, B)
    elif A.ndim == 2:
        C = numpy.dot(A, B)
    else:
        C = numpy.einsum('...ij,...jk->...ik', A, B)
    C = C.astype(dtype, copy=False)
    if A.ndim == 2:
        C_data = numpy.zeros((ldnz), dtype=dtype)
        nnz = len(numpy.where(C_row >= 0)[0])
        C_data[:nnz] = C[C_row[:nnz], C_col[:nnz]]
    else:
        nb = A.shape[0]
        C_data = numpy.zeros((nb, ldnz), dtype=dtype)
        for i in range(nb):
            nnz = len(numpy.where(C_row[i] >= 0)[0])
            C_data[i, :nnz] = C[i, C_row[i, :nnz], C_col[i, :nnz]]

    return C_data


def _coo_matmul_gradsp_gpu(A, B, C_row, C_col, dtype):
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
    _cupy_coo_matmul_gradsp()(nb, _m, _n, _k, ldnz, A, B, C_row, C_col, C_data,
                              size=nthreads)

    return C_data


def _cupy_coo_matmul_gradsp():
    return cuda.elementwise(
        'int32 nb, int32 _m, int32 _n, int32 _k, int32 nnz, \
         raw A _A, raw B _B, \
         raw T C_row, raw T C_col',
        'raw C C_data',
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
            val_C += static_cast<C>(val_A * val_B);
        }
        C_data[i_C] = val_C;
        ''',
        'coo_matmul_gradsp')


class CooMatMulGradSP(function_node.FunctionNode):

    def __init__(self, sp_row, sp_col, sp_shape, sp_order='other',
                 transa=False, transb=False, transc=False,
                 dtype=None):
        if sp_row.ndim != sp_col.ndim:
            raise ValueError('ndim of sp_row and sp_col must be the same.')
        if sp_row.ndim != 1 and sp_row.ndim != 2:
            raise ValueError('ndim of sp_row and sp_col must be one or two.')
        for i in range(sp_row.ndim):
            if sp_row.shape[i] != sp_col.shape[i]:
                msg = 'shape of sp_row and sp_col must be the same.'
                raise ValueError(msg)
        if len(sp_shape) != 2:
            raise ValueError('len(sp_shape) must be two.')
        self.sp_row = sp_row  # ((nb,) ldnz)
        self.sp_col = sp_col  # ((nb,) ldnz)
        self.sp_shape = sp_shape  # (_m, _n) when transc is False
        self.sp_order = sp_order
        self.transa = transa
        self.transb = transb
        self.transc = transc
        self.dtype = dtype

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        a_type, b_type = in_types
        # a_type.shape: ((nb,) _m, _k) when transa is False
        # b_type.shape: ((nb,) _k, _n) when transb is False
        a_m_axis, a_k_axis = -2, -1
        b_k_axis, b_n_axis = -2, -1
        sp_m_axis, sp_n_axis = -2, -1
        if self.transa:
            a_m_axis, a_k_axis = -1, -2
        if self.transb:
            b_k_axis, b_n_axis = -1, -2
        if self.transc:
            sp_m_axis, sp_n_axis = -1, -2
        type_check.expect(
            a_type.dtype.kind == 'f',
            b_type.dtype.kind == 'f',
            a_type.ndim >= 2,
            a_type.ndim <= 3,
            a_type.ndim == b_type.ndim,
            a_type.shape[a_m_axis] == self.sp_shape[sp_m_axis],
            b_type.shape[b_n_axis] == self.sp_shape[sp_n_axis],
            a_type.shape[a_k_axis] == b_type.shape[b_k_axis],
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
        c = _coo_matmul_gradsp(a, b, self.sp_row, self.sp_col, self.sp_shape,
                               self.transa, self.transb, self.transc,
                               self.dtype)
        return utils.force_array(c),

    def backward(self, indexes, grad_outputs):
        a, b = self.get_retained_inputs()
        g_sp, = grad_outputs
        ret = []
        if 0 in indexes:
            g_a = CooMatMul(self.sp_row, self.sp_col, self.sp_shape,
                            self.sp_order,
                            self.transc, not self.transb, self.transa,
                            dtype=a.dtype).apply((g_sp, b))[0]
            ret.append(g_a)
        if 1 in indexes:
            g_b = CooMatMul(self.sp_row, self.sp_col, self.sp_shape,
                            self.sp_order,
                            not self.transc, self.transa, not self.transb,
                            dtype=b.dtype).apply((g_sp, a))[0]
            ret.append(g_b)
        return ret


def sparse_matmul(a, b, transa=False, transb=False):
    """Computes the batched multiplication of sparse and dense matrix.

    The following use cases are supported:

        1. C (dense) = A (sparse) * B (dense)
        2. C (dense) = A (dense) * B (sparse)

    Args:
        a (~chainer.Variable or ~chainer.utils.CooMatrix): The left operand of
            matrix multiplication.
        b (~chainer.Variable or ~chainer.utils.CooMatrix): The right operand of
            matrix multiplication.
        transa (bool): If ``True``, each matrix in ``a`` will be transposed.
        transb (bool): If ``True``, each matrix in ``b`` will be transposed.

    Returns:
        ~chainer.Variable: Result of batched mat-mul.

    .. seealso::
        See :func:`~chainer.utils.to_coo` for how to construct a COO matrix
        from an array.

    .. note::
        Performance of this function on GPU can be improved by using the
        ``order`` argument of :class:`~chainer.utils.CooMatrix` when the sparse
        matrix is created.

    """
    if (isinstance(a, utils.CooMatrix) and
            isinstance(b, (chainer.Variable, numpy.ndarray, cuda.ndarray))):
        return CooMatMul(a.row, a.col, a.shape, a.order,
                         transa=transa,
                         transb=transb,
                         transc=False).apply((a.data, b))[0]
    elif (isinstance(a, (chainer.Variable, numpy.ndarray, cuda.ndarray)) and
          isinstance(b, utils.CooMatrix)):
        return CooMatMul(b.row, b.col, b.shape, b.order,
                         transa=not transb,
                         transb=not transa,
                         transc=True).apply((b.data, a))[0]
    else:
        msg = 'This combination of type of inputs is not supported.\n'
        msg += '    a: {}\n'.format(type(a))
        msg += '    b: {}\n'.format(type(b))
        raise ValueError(msg)
