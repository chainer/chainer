#include "chainerx/cuda/cuda_device.h"

#include <mutex>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "chainerx/cuda/cublas.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/cuda/cuda_set_device_scope.h"

namespace chainerx {
namespace cuda {

CudaDevice::~CudaDevice() {
    if (cublas_handle_ != nullptr) {
        // NOTE: CudaSetDeviceScope is not available because it may throw
        int orig_index{0};
        cudaGetDevice(&orig_index);
        cudaSetDevice(index());
        cublasDestroy(cublas_handle_);
        cudaSetDevice(orig_index);
    }
}

cublasHandle_t CudaDevice::cublas_handle() {
    if (cublas_handle_ == nullptr) {
        CudaSetDeviceScope scope{index()};
        CheckCublasError(cublasCreate(&cublas_handle_));
    }
    return cublas_handle_;
}

void CudaDevice::Synchronize() {
    CudaSetDeviceScope scope{index()};
    CheckCudaError(cudaDeviceSynchronize());
}

}  // namespace cuda
}  // namespace chainerx
