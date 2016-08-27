cimport cython

cdef extern from "cupy_cusparse.h":

    # cuSPARSE Helper Function
    Status cusparseCreate(Handle * handle)
    Status cusparseCreateMatDescr(MatDescr descr)
    Status cusparseDestroy(Handle handle)
    Status cusparseSetMatIndexBase(MatDescr descr, IndexBase base)
    Status cusparseSetMatType(MatDescr descr, MatrixType type)
    Status cusparseSetPointerMode(Handle handle, PointerMode mode)


    # cuSPARSE Level2 Function
    Status cusparseScsrmv(
        Handle handle, Operation transA, int m, int n, int nnz,
        const float *alpha, MatDescr descrA, const float *csrSortedValA,
        const int *csrSortedRowPtrA, const int *csrSortedColIndA,
        const float *x, const float *beta, float *y)


    # cuSPARSE Extra Function
    Status cusparseXcsrgeamNnz(
        Handle handle, int m, int n, const MatDescr descrA, int nnzA,
        const int *csrRowPtrA, const int *csrColIndA, const MatDescr descrB,
        int nnzB, const int *csrRowPtrB, const int *csrColIndB,
        const MatDescr descrC, int *csrRowPtrC, int *nnzTotalDevHostPtr)

    Status cusparseScsrgeam(
        Handle handle, int m, int n, const float *alpha, const MatDescr descrA,
        int nnzA, const float *csrValA, const int *csrRowPtrA,
        const int *csrColIndA, const float *beta, const MatDescr descrB,
        int nnzB, const float *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const MatDescr descrC, float *csrValC,
        int *csrRowPtrC, int *csrColIndC)

    Status cusparseXcsrgemmNnz(
        Handle handle, Operation transA, Operation transB, int m, int n, int k,
        const MatDescr descrA, const int nnzA, const int *csrRowPtrA,
        const int *csrColIndA, const MatDescr descrB, const int nnzB,
        const int *csrRowPtrB, const int *csrColIndB,
        const MatDescr descrC, int *csrRowPtrC, int *nnzTotalDevHostPtr)

    Status cusparseScsrgemm(
        Handle handle, Operation transA, Operation transB, int m, int n, int k,
        const MatDescr descrA, const int nnzA, const float *csrValA,
        const int *csrRowPtrA, const int *csrColIndA, const MatDescr descrB,
        const int nnzB, const float *csrValB, const int *csrRowPtrB,
        const int *csrColIndB, const MatDescr descrC, float *csrValC,
        const int *csrRowPtrC, int *csrColIndC)


    # cuSPARSE Format Convrsion
    Status cusparseScsr2dense(
        Handle handle, int m, int n, const MatDescr descrA,
        const float *csrSortedValA, const int *csrSortedRowPtrA,
        const int *csrSortedColIndA, float *A, int lda)


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


cpdef setPointerMode(size_t handle, int mode):
    status = cusparseSetPointerMode(<Handle>handle, <PointerMode>mode)
    check_status(status)


########################################
# cuSPARSE Level2 Function

cpdef scsrmv(
        size_t handle, int transA, int m, int n, int nnz,
        size_t alpha, size_t descrA, size_t csrSortedValA,
        size_t csrSortedRowPtrA, size_t csrSortedColIndA,
        size_t x, size_t beta, size_t y):
    status = cusparseScsrmv(
        <Handle>handle, <Operation>transA, m, n, nnz,
        <const float *>alpha, <MatDescr>descrA, <const float *>csrSortedValA,
        <const int *>csrSortedRowPtrA, <const int *>csrSortedColIndA,
        <const float *>x, <const float *>beta, <float *>y)
    check_status(status)


########################################
# cuSPARSE Extra Function

cpdef xcsrgeamNnz(
        size_t handle, int m, int n, size_t descrA, int nnzA,
        size_t csrRowPtrA, size_t csrColIndA, size_t descrB,
        int nnzB, size_t csrRowPtrB, size_t csrColIndB,
        size_t descrC, size_t csrRowPtrC, size_t nnzTotalDevHostPtr):
    status = cusparseXcsrgeamNnz(
        <Handle>handle, m, n, <const MatDescr>descrA, nnzA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const MatDescr>descrB, nnzB, <const int *>csrRowPtrB,
        <const int *>csrColIndB, <const MatDescr>descrC, <int *>csrRowPtrC,
        <int *>nnzTotalDevHostPtr)
    check_status(status)

cpdef scsrgeam(
        size_t handle, int m, int n, size_t alpha, size_t descrA,
        int nnzA, size_t csrValA, size_t csrRowPtrA,
        size_t csrColIndA, size_t beta, size_t descrB,
        int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC):
    status = cusparseScsrgeam(
        <Handle>handle, m, n, <const float *>alpha,
        <const MatDescr>descrA, nnzA, <const float *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA, <const float *>beta,
        <const MatDescr>descrB, nnzB, <const float *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <float *>csrValC, <int *>csrRowPtrC,
        <int *>csrColIndC)
    check_status(status)


cpdef xcsrgemmNnz(
        size_t handle, int transA, int transB, int m, int n, int k,
        size_t descrA, int nnzA, size_t csrRowPtrA,
        size_t csrColIndA, size_t descrB, int nnzB,
        size_t csrRowPtrB, size_t csrColIndB,
        size_t descrC, size_t csrRowPtrC, size_t nnzTotalDevHostPtr):
    status = cusparseXcsrgemmNnz(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k,
        <const MatDescr>descrA, nnzA, <const int *>csrRowPtrA,
        <const int *>csrColIndA, <const MatDescr>descrB, nnzB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <int *>csrRowPtrC, <int *>nnzTotalDevHostPtr)
    check_status(status)


cpdef scsrgemm(
        size_t handle, int transA, int transB, int m, int n, int k,
        size_t descrA, const int nnzA, size_t csrValA,
        size_t csrRowPtrA, size_t csrColIndA, size_t descrB,
        const int nnzB, size_t csrValB, size_t csrRowPtrB,
        size_t csrColIndB, size_t descrC, size_t csrValC,
        size_t csrRowPtrC, size_t csrColIndC):
    status = cusparseScsrgemm(
        <Handle>handle, <Operation>transA, <Operation>transB, m, n, k,
        <const MatDescr>descrA, nnzA, <const float *>csrValA,
        <const int *>csrRowPtrA, <const int *>csrColIndA,
        <const MatDescr>descrB, nnzB, <const float *>csrValB,
        <const int *>csrRowPtrB, <const int *>csrColIndB,
        <const MatDescr>descrC, <float *>csrValC,
        <const int *>csrRowPtrC, <int *>csrColIndC)
    check_status(status)


########################################
# cuSPARSE Format Convrsion

cpdef scsr2dense(
        size_t handle, int m, int n, size_t descrA,
        size_t csrSortedValA, size_t csrSortedRowPtrA,
        size_t csrSortedColIndA, size_t A, int lda):
    status = cusparseScsr2dense(
        <Handle>handle, m, n, <MatDescr>descrA,
        <const float *>csrSortedValA, <const int *>csrSortedRowPtrA,
        <const int *>csrSortedColIndA, <float *>A, lda)
    check_status(status)
