"""
Wrapper for NCCL: Optimized primiteive for collective multi-GPU communication
"""
cimport cpython
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
        ncclUnhandledCudaError
        ncclSystemError
        ncclInternalError
        ncclInvalidDevicePointer
        ncclInvalidRank
        ncclUnsupportedDeviceCount
        ncclDeviceNotFound
        ncclInvalidDeviceIndex
        ncclLibWrapperNotSet
        ncclCudaMallocFailed
        ncclRankMismatch
        ncclInvalidArgument
        ncclInvalidType
        ncclInvalidOperation
        nccl_NUM_RESULTS
    ctypedef enum ncclRedOp_t:
        ncclSum
        ncclProd
        ncclMax
        ncclMin
        nccl_NUM_OPS
    ctypedef enum ncclDataType_t:
        ncclChar
        ncclInt
        # ncclHalf
        ncclFloat
        ncclDouble
        ncclInt64
        ncclUint64
        nccl_NUM_TYPES

    int ncclCommInitAll(ncclComm_t* comm, int ndev, int *devlist)
    void ncclCommDestroy(ncclComm_t comm)
    int ncclAllReduce(const void* sendbuff, void* recvbuff, int count,
                      ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm,
                      driver.Stream stream);

cdef struct comm_pointers:
    size_t ptr[64]

class NcclCommunicator(object):

    def __init__(self, int ndev, list devlist):
        self.ndev = ndev
        self.devlist = devlist

        cdef int* _devlist = <int*>malloc(ndev * sizeof(int))
        for i in range(ndev):
            _devlist[i] = devlist[i]
            print("[nccl.pyx, __init__()] devlist[{}]:{}".format(i, devlist[i]))

        cdef ncclComm_t* _comms = <ncclComm_t*>malloc(ndev * sizeof(ncclComm_t))
        ret = ncclCommInitAll(_comms, ndev, _devlist)
        print("[nccl.pyx, __init__()] ret:{}".format(ret))

        cdef comm_pointers comm_ptrs
        for i in range(ndev):
            comm_ptrs.ptr[i] = <size_t>_comms[i]
            print("[nccl.pyx, __init__()] comm_ptrs.ptr[{}]: {}".format(i, comm_ptrs.ptr[i]))

        self.comm_ptrs = comm_ptrs

        free(_devlist)
        free(_comms)

    def destroy(self):
        cdef comm_pointers comm_ptrs = self.comm_ptrs
        for i in range(self.ndev):
            print("[nccl.pyx, destroy()] comm_ptrs.ptr[{}]: {}".format(i, comm_ptrs.ptr[i]))
            ncclCommDestroy(<ncclComm_t> comm_ptrs.ptr[i])

    def AllReduce(self, rank, size_t sendbuf, size_t recvbuf, 
                  count, datatype, op, size_t stream):
        cdef comm_pointers comm_ptrs = self.comm_ptrs
        cdef ncclComm_t comm = <ncclComm_t>comm_ptrs.ptr[<int>rank]
        ncclAllReduce( <void*>sendbuf, <void*>recvbuf, <int>count,
                       <ncclDataType_t>datatype, <ncclRedOp_t>op, comm, <driver.Stream>stream )
