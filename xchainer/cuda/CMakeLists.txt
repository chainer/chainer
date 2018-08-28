install(FILES
    cublas.h
    cuda.h
    cuda_backend.h
    cuda_conv.h
    cuda_device.h
    cuda_runtime.h
    cudnn.h
    elementwise.cuh
    memory_pool.h
    reduce.cuh
    DESTINATION include/xchainer/cuda
    )

CUDA_ADD_LIBRARY(xchainer_cuda STATIC
    cublas.cc
    cuda.cc
    cuda_conv.cc
    cuda_device.cc
    cuda_device/activation.cu
    cuda_device/arithmetic.cu
    cuda_device/batch_norm.cc
    cuda_device/conv.cc
    cuda_device/copy.cu
    cuda_device/comparison.cu
    cuda_device/dot.cc
    cuda_device/exp_log.cu
    cuda_device/fill.cu
    cuda_device/indexing.cu
    cuda_device/memory.cc
    cuda_device/misc.cu
    cuda_device/pool.cu
    cuda_device/reduction.cu
    cuda_backend.cc
    cuda_runtime.cc
    cudnn.cc
    memory_pool.cc
    )

CUDA_ADD_CUBLAS_TO_TARGET(xchainer_cuda)

include_directories(${CUDNN_INCLUDE_DIRS})
target_link_libraries(xchainer_cuda ${CUDNN_LIBRARIES})

if(${XCHAINER_BUILD_TEST})
    add_executable(xchainer_cuda_test
        cuda_backend_test.cc
        cuda_conv_test.cc
        cuda_device_test.cc
        cuda_runtime_test.cc
        memory_pool_test.cc
        )
    target_link_libraries(xchainer_cuda_test
        xchainer
        xchainer_cuda
        xchainer_testing
        gtest)
    add_test(NAME xchainer_cuda_test COMMAND xchainer_cuda_test)
endif()
