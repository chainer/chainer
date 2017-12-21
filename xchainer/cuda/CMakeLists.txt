add_subdirectory(libs)

CUDA_ADD_LIBRARY(xchainer_cuda STATIC
    hello.cu
    libs/cuda_runtime.cc)
