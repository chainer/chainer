# distutils: language = c++

"""Thin wrapper of cuda profiler."""
from cupy.cuda cimport runtime


cdef extern from "cupy_cuda.h":
    runtime.Error cudaProfilerInitialize(const char *configFile,
                                         const char *outputFile,
                                         int outputMode) nogil
    runtime.Error cudaProfilerStart() nogil
    runtime.Error cudaProfilerStop() nogil


cpdef void initialize(str config_file,
                      str output_file,
                      int output_mode) except *:
    """Initialize the CUDA profiler.

    This function initialize the CUDA profiler. See the CUDA document for
    detail.

    Args:
        config_file (str): Name of the configuration file.
        output_file (str): Name of the coutput file.
        output_mode (int): ``cupy.cuda.profiler.cudaKeyValuePair`` or
            ``cupy.cuda.profiler.cudaCSV``.
    """
    cdef bytes b_config_file = config_file.encode()
    cdef bytes b_output_file = output_file.encode()
    cdef const char* b_config_file_ptr = <const char*>b_config_file
    cdef const char* b_output_file_ptr = <const char*>b_output_file
    with nogil:
        status = cudaProfilerInitialize(b_config_file_ptr,
                                        b_output_file_ptr,
                                        <OutputMode>output_mode)
    runtime.check_status(status)


cpdef void start() except *:
    """Enable profiling.

    A user can enable CUDA profiling. When an error occurs, it raises an
    exception.

    See the CUDA document for detail.
    """
    with nogil:
        status = cudaProfilerStart()
    runtime.check_status(status)


cpdef void stop() except *:
    """Disable profiling.

    A user can disable CUDA profiling. When an error occurs, it raises an
    exception.

    See the CUDA document for detail.
    """
    with nogil:
        status = cudaProfilerStop()
    runtime.check_status(status)
