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
    ctypedef enum ncclResult_t:
        ncclSuccess
    ctypedef enum ncclRedOp_t:
        pass
    ctypedef enum ncclDataType_t:
        pass

    const char* ncclGetErrorString(ncclResult_t result)
    ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, int *devlist)
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


cdef struct comm_info:
    size_t ptr


class NcclCommunicator(object):

    def __init__(self, int ndev, list devlist):
        self.ndev = ndev
        self.devlist = devlist

        cdef int* _devlist = <int*>malloc(ndev * sizeof(int))
        for i in range(ndev):
            _devlist[i] = devlist[i]
            print("[nccl.pyx, __init__()] devlist[{}]:{}".format(i, devlist[i]))

        cdef ncclComm_t* _comms = <ncclComm_t*>malloc(ndev * sizeof(ncclComm_t))
        status = ncclCommInitAll(_comms, ndev, _devlist)
        check_status(status)
            
        self.comms = []
        cdef comm_info comm
        for i in range(ndev):
            comm.ptr = <size_t>_comms[i]
            print("[nccl.pyx, __init__()] comm.ptr: {}".format(comm.ptr))
            self.comms.append(comm)

        free(_devlist)
        free(_comms)

    def destroy(self):
        cdef comm_info comm
        for i in range(self.ndev):
            comm = self.comms[i]
            print("[nccl.pyx, destroy()] comm.ptr: {}".format(comm.ptr))
            ncclCommDestroy(<ncclComm_t>comm.ptr)

    def allReduce(self, int rank, size_t sendbuf, size_t recvbuf, 
                  int count, int datatype, int op, size_t stream):
        cdef comm_info comm = self.comms[rank]
        status = ncclAllReduce( <void*>sendbuf, <void*>recvbuf, count,
                                <ncclDataType_t>datatype, <ncclRedOp_t>op,
                                <ncclComm_t>comm.ptr, <driver.Stream>stream )
        check_status(status)


class NcclCommunicatorMP(object):

    def __init__(self, int ndev, commId, int rank):
