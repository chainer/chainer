import cupy
from cupy.cuda import runtime


class IpcMemoryHandle(object):

    """CUDA interprocess memory handle

    Args:
        array (cupy.ndarray): cupy ndarray to be shared among multiple processes

    Attributes:
        mh_data (list): data of memory handle to be created by cudaIpcGetMemHandle()

        dtype (numpy.dtype): Dtype object of element type
        size (int): Number of elements
        device_id (int): Device ID
        devptr (int): Pointer to device memory which is initiated by another process
    """

    def __init__(self, array):
        self.mh_data = runtime.ipcGetMemHandle(array.data.ptr)
        self.dtype = array.dtype
        self.size = array.size
        self.device_id = array.data.device.id
        self.devptr = 0  # NULL
        #print("[ipc.py, __init__()] array.data.ptr:{}".format(array.data.ptr))
        #print("[ipc.py, __init__()] self.device_id:{}".format(self.device_id))

    def open(self):
        current_device_id = runtime.getDevice()
        runtime.setDevice(self.device_id)
        self.devptr = runtime.ipcOpenMemHandle(self.mh_data)
        #print("[ipc.py, open()] self.devptr:{}".format(self.devptr))
        #print("[ipc.py, open()] self.device_id:{}".format(self.device_id))
        array = cupy.ndarray(shape=(self.size,), dtype=self.dtype,
                             devptr=self.devptr)
        runtime.setDevice(current_device_id)
        return array

    def close(self):
        runtime.ipcCloseMemHandle(self.devptr)
