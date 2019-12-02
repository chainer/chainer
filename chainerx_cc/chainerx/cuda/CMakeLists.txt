install(FILES
    cublas.h
    cuda.h
    cuda_backend.h
    cuda_conv.h
    cuda_device.h
    cuda_set_device_scope.h
    cuda_runtime.h
    cudnn.h
    cusolver.h
    data_type.cuh
    elementwise.cuh
    float16.cuh
    kernel_regist.h
    memory_pool.h
    numeric.cuh
    numeric_limits.cuh
    reduce.cuh
    DESTINATION include/chainerx/cuda
    )

CUDA_ADD_LIBRARY(chainerx_cuda STATIC
    cublas.cc
    cuda.cc
    cuda_conv.cc
    cuda_device.cc
    cuda_device/arithmetic.cu
    cuda_device/batch_norm.cc
    cuda_device/binary.cu
    cuda_device/conv.cc
    cuda_device/copy.cu
    cuda_device/logic.cu
    cuda_device/dot.cu
    cuda_device/exp_log.cu
    cuda_device/fill.cu
    cuda_device/hyperbolic.cu
    cuda_device/indexing.cu
    cuda_device/linalg.cu
    cuda_device/memory.cc
    cuda_device/misc.cu
    cuda_device/pool.cu
    cuda_device/rnn.cu
    cuda_device/reduction.cu
    cuda_device/rounding.cu
    cuda_device/statistics.cu
    cuda_device/trigonometric.cu
    cuda_backend.cc
    cuda_runtime.cc
    cudnn.cc
    cusolver.cc
    memory_pool.cc
    )

CUDA_ADD_CUBLAS_TO_TARGET(chainerx_cuda)

target_link_libraries(chainerx_cuda ${CUDNN_LIBRARIES} ${CUDA_cusolver_LIBRARY})

if(${CHAINERX_BUILD_TEST})
    CUDA_ADD_EXECUTABLE(chainerx_cuda_test
        cuda_backend_test.cc
        cuda_conv_test.cc
        cuda_set_device_scope_test.cc
        cuda_device_test.cc
        cuda_runtime_test.cc
        memory_pool_test.cc
        )
    target_link_libraries(chainerx_cuda_test
        chainerx
        chainerx_test_main
        # gtest is linked via chainerx_testing.
        )
    add_test(NAME chainerx_cuda_test COMMAND chainerx_cuda_test)
endif()
