"""Thin wrapper of CUSOLVER."""
cimport cython


###############################################################################
# Extern
###############################################################################

cdef extern from 'cupy_cuda.h':
    # Context
    int cusolverDnCreate(Handle* handle)
    int cusolverDnDestroy(Handle handle)

    # Stream
    int cusolverDnGetStream(Handle handle, Stream* streamId)
    int cusolverDnSetStream(Handle handle, Stream streamId)

    # Linear Equations
    int cusolverDnSpotrf(Handle handle, FillMode uplo, int n, float* A,
                         int lda, float* Workspace, int Lwork, int* devInfo)
    int cusolverDnDpotrf(Handle handle, FillMode uplo, int n, double *A,
                         int lda, double* Workspace, int Lwork, int* devInfo)
    int cusolverDnSgetrs(Handle handle, Operation trans,
                         int n, int nrhs, const float* A, int lda,
                         const int* devIpiv, float* B, int ldb, int* devInfo)
    int cusolverDnDgetrs(Handle handle, Operation trans,
                         int n, int nrhs, const double* A, int lda,
                         const int* devIpiv, double* B, int ldb, int* devInfo)
    int cusolverDnSgesvd(Handle handle, char jobu, char jobvt, int m, int n,
                         float* A, int lda, float* S, float* U, int ldu,
                         float* VT, int ldvt, float* Work, int Lwork,
                         float* rwork, int* devInfo)
    int cusolverDnDgesvd(Handle handle, char jobu, char jobvt, int m, int n,
                         double* A, int lda, double* S, double* U, int ldu,
                         double* VT, int ldvt, double* Work, int Lwork,
                         double* rwork, int* devInfo)

###############################################################################
# Error handling
###############################################################################

cdef dict STATUS = {
    0: 'CUSOLVER_STATUS_SUCCESS',
    1: 'CUSOLVER_STATUS_NOT_INITIALIZED',
    2: 'CUSOLVER_STATUS_ALLOC_FAILED',
    3: 'CUSOLVER_STATUS_INVALID_VALUE',
    4: 'CUSOLVER_STATUS_ARCH_MISMATCH',
    5: 'CUSOLVER_STATUS_MAPPING_ERROR',
    6: 'CUSOLVER_STATUS_EXECUTION_FAILED',
    7: 'CUSOLVER_STATUS_INTERNAL_ERROR',
    8: 'CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED',
    9: 'CUSOLVER_STATUS_NOT_SUPPORTED',
    10: 'CUSOLVER_STATUS_ZERO_PIVOT',
    11: 'CUSOLVER_STATUS_INVALID_LICENSE',
}


class CUSOLVERError(RuntimeError):

    def __init__(self, status):
        self.status = status
        super(CUSOLVERError, self).__init__(STATUS[status])


@cython.profile(False)
cpdef inline check_status(int status):
    if status != 0:
        raise CUSOLVERError(status)


###############################################################################
# Context
###############################################################################

cpdef size_t create() except *:
    cdef Handle handle
    status = cusolverDnCreate(&handle)
    check_status(status)
    return <size_t>handle


cpdef void destroy(size_t handle) except *:
    status = cusolverDnDestroy(<Handle>handle)
    check_status(status)

###############################################################################
# Stream
###############################################################################

cpdef setStream(size_t handle, size_t stream):
    status = cusolverDnSetStream(<Handle>handle, <Stream>stream)
    check_status(status)


cpdef size_t getStream(size_t handle) except *:
    cdef Stream stream
    status = cusolverDnGetStream(<Handle>handle, &stream)
    check_status(status)
    return <size_t>stream

###############################################################################
# dense LAPACK Functions
###############################################################################

cpdef int spotrf(size_t handle, int uplo, int n, size_t A, int lda,
                 size_t Workspace, int Lwork, size_t devInfo):
    cdef int result
    status = cusolverDnSpotrf(
        <Handle>handle, <FillMode>uplo, n, <float*>A,
        lda, <float*>Workspace, Lwork, <int*>devInfo)
    check_status(result)

cpdef int dpotrf(size_t handle, int uplo, int n, size_t A, int lda,
                 size_t Workspace, int Lwork, size_t devInfo):
    cdef int result
    status = cusolverDnDpotrf(
        <Handle>handle, <FillMode>uplo, n, <double*>A,
        lda, <double*>Workspace, Lwork, <int*>devInfo)
    check_status(result)

cpdef int sgetrs(size_t handle, int trans, int n, int nrhs,
                 size_t A, int lda, size_t devIpiv,
                 size_t B, int ldb, size_t devInfo):
    cdef int result
    status = cusolverDnSgetrs(
        <Handle>handle, <Operation>trans, n, nrhs,
        <const float*> A, lda, <const int*>devIpiv,
        <float*>B, ldb, <int*> devInfo)
    check_status(result)

cpdef int dgetrs(size_t handle, int trans, int n, int nrhs,
                 size_t A, int lda, size_t devIpiv,
                 size_t B, int ldb, size_t devInfo):
    cdef int result
    status = cusolverDnDgetrs(
        <Handle>handle, <Operation>trans, n, nrhs,
        <const double*> A, lda, <const int*>devIpiv,
        <double*>B, ldb, <int*> devInfo)
    check_status(result)

cpdef int sgesvd(size_t handle, char jobu, char jobvt, int m, int n, size_t A,
                 int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
                 size_t Work, int Lwork, size_t rwork, size_t devInfo):
    cdef int result
    status = cusolverDnSgesvd(
        <Handle>handle, jobu, jobvt, m, n, <float*>A,
        lda, <float*>S, <float*>U, ldu, <float*>VT, ldvt,
        <float*>Work, Lwork, <float*>rwork, <int*>devInfo)
    check_status(result)

cpdef int dgesvd(size_t handle, char jobu, char jobvt, int m, int n, size_t A,
                 int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
                 size_t Work, int Lwork, size_t rwork, size_t devInfo):
    cdef int result
    status = cusolverDnDgesvd(
        <Handle>handle, jobu, jobvt, m, n, <double*>A,
        lda, <double*>S, <double*>U, ldu, <double*>VT, ldvt,
        <double*>Work, Lwork, <double*>rwork, <int*>devInfo)
    check_status(result)
