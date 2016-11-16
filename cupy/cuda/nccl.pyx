"""
Wrapper for NCCL: Optimized primiteive for collective multi-GPU communication
"""
cimport cython

from cupy.cuda cimport driver
from libc.stdlib cimport malloc
from libc.stdlib cimport free

cdef extern from "nccl.h":
    ctypedef struct ncclComm:
        pass
    ctypedef ncclComm* ncclComm_t
    cdef enum:
        NCCL_UNIQUE_ID_BYTES = 128
    ctypedef struct ncclUniqueId:
        char internal[NCCL_UNIQUE_ID_BYTES]
    ctypedef enum ncclResult_t:
        ncclSuccess
    ctypedef enum ncclRedOp_t:
        pass
    ctypedef enum ncclDataType_t:
        pass

    const char* ncclGetErrorString(ncclResult_t result)
    ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId)
    ncclResult_t ncclCommInitRank(ncclComm_t* comm, int ndev, ncclUniqueId commId, int rank)
    void ncclCommDestroy(ncclComm_t comm)
    ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, int count,
                               ncclDataType_t datatype, ncclRedOp_t op,
                               ncclComm_t comm, driver.Stream stream)


cdef dict STATUS = {
    0: 'NCCL_STATUS_SUCCESS',
    1: 'NCCL_STATUS_UNHANDLED_CUDA_ERROR',
    2: 'NCCL_STATUS_SYSTEM_ERROR',
    3: 'NCCL_STATUS_INTERNAL_ERROR',
    4: 'NCCL_STATUS_INVALID_DEVICE_POINTER',
    5: 'NCCL_STATUS_INVALID_RANK',
    6: 'NCCL_STATUS_UNSUPPORTED_DEVICE_COUNT',
    7: 'NCCL_STATUS_DEVICE_NOT_FOUND',
    8: 'NCCL_STATUS_INVALID_DEVICE_INDEX',
    9: 'NCCL_STATUS_LIB_WRAPPER_NOT_SET',
    10: 'NCCL_STATUS_CUDA_MALLOC_FAILED',
    11: 'NCCL_STATUS_RANK_MISMATCH',
    12: 'NCCL_STATUS_INVALID_ARGUMENT',
    13: 'NCCL_STATUS_INVALID_TYPE',
    14: 'NCCL_STATUS_INVALID_OPERATION',
}


class NcclError(RuntimeError):

    def __init__(self, int status):
        self.status = status
        msg = ncclGetErrorString(<ncclResult_t>status)
        super(NcclError, self).__init__('%s: %s' % (STATUS[status], msg))


@cython.profile(False)
cpdef inline check_status(ncclResult_t status):
    if status != ncclSuccess:
        raise NcclError(status)


class NcclCommunicatorId(object):

    def __init__(self):
        cdef ncclUniqueId uniqueId
        status = ncclGetUniqueId(&uniqueId)
        check_status(status)
        self.data = []
        for i in range(NCCL_UNIQUE_ID_BYTES):
            self.data.append(<char>uniqueId.internal[i])

        
cdef struct comm_info:
    size_t ptr


class NcclCommunicator(object):

    def __init__(self, int ndev, commId, int rank):
        cdef ncclUniqueId _uniqueId
        for i in range(NCCL_UNIQUE_ID_BYTES):
            _uniqueId.internal[i] = commId.data[i]
        cdef ncclComm_t _comm
        status = ncclCommInitRank(&_comm, ndev, _uniqueId, rank)
        check_status(status)
            
        cdef comm_info _ci
        _ci.ptr = <size_t>_comm
        #print("[nccl.pyx, __init__()] _ci.ptr: {}".format(_ci.ptr))
        self.ci = _ci

    def destroy(self):
        cdef comm_info _ci = self.ci
        #print("[nccl.pyx, destroy()] _ci.ptr: {}".format(_ci.ptr))
        ncclCommDestroy(<ncclComm_t>_ci.ptr)

    def allReduce(self, size_t sendbuf, size_t recvbuf, 
                  int count, int datatype, int op, size_t stream):
        cdef comm_info _ci = self.ci
        #print("[nccl.pyx, allReduce()] _ci.ptr: {}".format(_ci.ptr))
        status = ncclAllReduce( <void*>sendbuf, <void*>recvbuf, count,
                                <ncclDataType_t>datatype, <ncclRedOp_t>op,
                                <ncclComm_t>_ci.ptr, <driver.Stream>stream )
        check_status(status)
