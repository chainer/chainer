CUDA_ADD_LIBRARY(xchainer_cuda STATIC
    cublas.cc
    cuda_device.cu
    cuda_backend.cc
    cuda_runtime.cc)

CUDA_ADD_CUBLAS_TO_TARGET(xchainer_cuda)

if(${ENABLE_TEST})
    add_executable(xchainer_cuda_test
        cuda_backend_test.cc
        cuda_device_test.cc
        cuda_runtime_test.cc
        )
    target_link_libraries(xchainer_cuda_test
        xchainer
        xchainer_cuda
        xchainer_testing
        gtest)
    add_test(NAME xchainer_cuda_test COMMAND xchainer_cuda_test)
endif()
