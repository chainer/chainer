CUDA_ADD_LIBRARY(xchainer_cuda STATIC
    cuda_device.cu
    cuda_backend.cc
    cuda_runtime.cc)

if(${ENABLE_TEST})
  add_executable(xchainer_cuda_test
      cuda_backend_test.cc
      cuda_device_test.cc
  )
  target_link_libraries(xchainer_cuda_test
      xchainer
      xchainer_cuda
      gtest_main)
  add_test(NAME xchainer_cuda_test COMMAND xchainer_cuda_test)
endif()
