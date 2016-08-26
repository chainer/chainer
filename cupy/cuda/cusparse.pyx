cimport cython

cdef extern from "cupy_cusparse.h":

    # cuSPARSE Helper Function
    Status cusparseCreate(Handle * handle)
    Status cusparseCreateMatDescr(MatDescr descr)
    Status cusparseDestroy(Handle handle)
    Status cusparseSetMatIndexBase(MatDescr descr, IndexBase base)
    Status cusparseSetMatType(MatDescr descr, MatrixType type)


cdef dict STATUS = {
    0: 'CUSPARSE_STATUS_SUCCESS',
    1: 'CUSPARSE_STATUS_NOT_INITIALIZED',
    2: 'CUSPARSE_STATUS_ALLOC_FAILED',
    3: 'CUSPARSE_STATUS_INVALID_VALUE',
    4: 'CUSPARSE_STATUS_ARCH_MISMATCH',
    5: 'CUSPARSE_STATUS_MAPPING_ERROR',
    6: 'CUSPARSE_STATUS_EXECUTION_FAILED',
    7: 'CUSPARSE_STATUS_INTERNAL_ERROR',
    8: 'CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED',
    9: 'CUSPARSE_STATUS_ZERO_PIVOT',
}


class CuSparseError(RuntimeError):

    def __init__(self, int status):
        self.status = status
        super(CuSparseError, self).__init__('%s' % (STATUS[status]))
        # msg = cudnnGetErrorString(<Status>status)
        #super(CuSparseError, self).__init__('%s: %s' % (STATUS[status], msg))


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CuSparseError(status)


########################################
# cuSPARSE Helper Function

cpdef size_t create() except *:
    cdef Handle handle
    status = cusparseCreate(& handle)
    check_status(status)
    return <size_t >handle


cpdef createMatDescr():
    cdef MatDescr desc
    status = cusparseCreateMatDescr(& desc)
    check_status(status)
    return <size_t>desc


cpdef destroy(size_t handle):
    status = cusparseDestroy(<Handle >handle)
    check_status(status)


cpdef setMatIndexBase(size_t descr, base):
    status = cusparseSetMatIndexBase(<MatDescr>descr, base)
    check_status(status)


cpdef setMatType(size_t descr, typ):
    status = cusparseSetMatType(<MatDescr>descr, typ)
    check_status(status)
