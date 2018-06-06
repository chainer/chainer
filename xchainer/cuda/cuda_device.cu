#include "xchainer/cuda/cuda_device.h"

#include <cublas_v2.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>

#include <cuda_runtime.h>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/backend_util.h"
#include "xchainer/cuda/cast.cuh"
#include "xchainer/cuda/cublas.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/cuda/elementwise.cuh"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/enum.h"
#include "xchainer/error.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/native/native_device.h"
#include "xchainer/routines/connection.h"
#include "xchainer/routines/creation.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace cuda {

CudaDevice::~CudaDevice() {
    if (cublas_handle_) {
        cudaSetDevice(index());
        cublasDestroy(cublas_handle_);
    }
}

cublasHandle_t CudaDevice::cublas_handle() {
    if (!cublas_handle_) {
        CheckCudaError(cudaSetDevice(index()));
        CheckCublasError(cublasCreate(&cublas_handle_));
    }
    return cublas_handle_;
}

void CudaDevice::Synchronize() {
    CheckCudaError(cudaSetDevice(index()));
    CheckCudaError(cudaDeviceSynchronize());
}

}  // namespace cuda
}  // namespace xchainer
