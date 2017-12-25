CUDA_ADD_LIBRARY(xchainer_cuda STATIC
    array_math.cu
    cuda_runtime.cc
    hello.cu)

if(${ENABLE_TEST})
    add_executable(xchainer_cuda_test
        hello_test.cc
        )
    target_link_libraries(xchainer_cuda_test
        xchainer
        gtest_main)
    add_test(NAME xchainer_cuda_test COMMAND xchainer_cuda_test)
endif()
