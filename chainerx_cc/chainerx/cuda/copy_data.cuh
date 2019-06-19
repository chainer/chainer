#pragma once
#include "chainerx/cuda/cuda.h"
#include "chainerx/cuda/cuda_runtime.h"
#include <cuda.h>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>


namespace chainerx {
namespace cuda {

__global__ void initGPUData_ker(float *data, int numElements, float* value) {
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   if (tid < numElements) {
      data[tid] = value[tid];
   }
}

void initGPUData(float *data, int numElements, float* value) {
   dim3 gridDim;
   dim3 blockDim;
   
   blockDim.x = 1024;
   gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;
   
   initGPUData_ker <<< gridDim, blockDim >>> (data, numElements, value);
}

}
}