"""Thin wrapper of CUSOLVER."""


###############################################################################
# Types
###############################################################################

from cupy.cuda.driver cimport Stream

cdef extern from *:
    ctypedef void* Handle 'cusolverDnHandle_t'

    ctypedef int Operation 'cublasOperation_t'
    ctypedef int SideMode 'cublasSideMode_t'
    ctypedef int FillMode 'cublasFillMode_t'

###############################################################################
# Enum
###############################################################################

cpdef enum:
    CUBLAS_OP_N = 0
    CUBLAS_OP_T = 1
    CUBLAS_OP_C = 2

###############################################################################
# Context
###############################################################################

cpdef size_t create() except *
cpdef void destroy(size_t handle) except *

###############################################################################
# Stream
###############################################################################

cpdef setStream(size_t handle, size_t stream)
cpdef size_t getStream(size_t handle) except *

###############################################################################
# dense LAPACK Functions
###############################################################################

cpdef int spotrf(size_t handle, int uplo, int n, size_t A, int lda,
                 size_t Workspace, int Lwork, size_t devInfo)
cpdef int dpotrf(size_t handle, int uplo, int n, size_t A, int lda,
                 size_t Workspace, int Lwork, size_t devInfo)

cpdef int spotrs(size_t handle, int uplo, int n, int nrhs,
                 size_t A, int lda, size_t B, int ldb, size_t devInfo)
cpdef int dpotrs(size_t handle, int uplo, int n, int nrhs,
                 size_t A, int lda, size_t B, int ldb, size_t devInfo)

cpdef int sgetrf(size_t handle, int m, int n, size_t A, int lda,
                 size_t Workspace, size_t devIpiv, size_t devInfo)
cpdef int dgetrf(size_t handle, int m, int n, size_t A, int lda,
                 size_t Workspace, size_t devIpiv, size_t devInfo)

cpdef int sgetrs(size_t handle, int trans, int n, int nrhs,
                 size_t A, int lda, size_t devIpiv,
                 size_t B, int ldb, size_t devInfo)
cpdef int dgetrs(size_t handle, int trans, int n, int nrhs,
                 size_t A, int lda, size_t devIpiv,
                 size_t B, int ldb, size_t devInfo)

cpdef int sgeqrf(size_t handle, int m, int n, size_t A, int lda,
                 size_t TAU, size_t Workspace, int Lwork, size_t devInfo)
cpdef int dgeqrf(size_t handle, int m, int n, size_t A, int lda,
                 size_t TAU, size_t Workspace, int Lwork, size_t devInfo)

cpdef int sormqr(size_t handle, int side, int trans,
                 int m, int n, int k, size_t A, int lda, size_t tau,
                 size_t C, int ldc, size_t work, int lwork, size_t devInfo)
cpdef int dormqr(size_t handle, int side, int trans,
                 int m, int n, int k, size_t A, int lda, size_t tau,
                 size_t C, int ldc, size_t work, int lwork, size_t devInfo)

cpdef int ssytrf(size_t handle, int uplo, int n, size_t A, int lda,
                 size_t ipiv, size_t work, int lwork, size_t devInfo)
cpdef int dsytrf(size_t handle, int uplo, int n, size_t A, int lda,
                 size_t ipiv, size_t work, int lwork, size_t devInfo)

cpdef int sgebrd(size_t handle, int m, int n, size_t A, int lda,
                 size_t D, size_t E, size_t TAUQ, size_t TAUP,
                 size_t Work, int Lwork, size_t devInfo)
cpdef int dgebrd(size_t handle, int m, int n, size_t A, int lda,
                 size_t D, size_t E, size_t TAUQ, size_t TAUP,
                 size_t Work, int Lwork, size_t devInfo)

cpdef int sgesvd(size_t handle, char jobu, char jobvt, int m, int n, size_t A,
                 int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
                 size_t Work, int Lwork, size_t rwork, size_t devInfo)
cpdef int dgesvd(size_t handle, char jobu, char jobvt, int m, int n, size_t A,
                 int lda, size_t S, size_t U, int ldu, size_t VT, int ldvt,
                 size_t Work, int Lwork, size_t rwork, size_t devInfo)
