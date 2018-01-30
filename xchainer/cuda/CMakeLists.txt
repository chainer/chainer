CUDA_ADD_LIBRARY(xchainer_cuda STATIC
    array_fill.cu
    array_math.cu
    cuda_backend.cc
    cuda_runtime.cc)
