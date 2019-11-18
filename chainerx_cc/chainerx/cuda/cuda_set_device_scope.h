#pragma once

#include <cuda_runtime.h>

#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/error.h"
#include "chainerx/macro.h"

namespace chainerx {
namespace cuda {

class CudaSetDeviceScope {
public:
    explicit CudaSetDeviceScope(int index) : index_{index} {
        CheckCudaError(cudaGetDevice(&orig_index_));
        CheckCudaError(cudaSetDevice(index_));
    }

    ~CudaSetDeviceScope() {
        if (CHAINERX_DEBUG) {
            cudaError_t status = cudaSetDevice(orig_index_);
            CHAINERX_ASSERT(status == cudaSuccess);
        } else {
            cudaSetDevice(orig_index_);
        }
    }

    CudaSetDeviceScope(const CudaSetDeviceScope&) = delete;
    CudaSetDeviceScope& operator=(const CudaSetDeviceScope&) = delete;
    CudaSetDeviceScope& operator=(CudaSetDeviceScope&&) = delete;
    CudaSetDeviceScope(CudaSetDeviceScope&& other) = delete;

    int index() const { return index_; }

private:
    int index_{};
    int orig_index_{};
};

}  // namespace cuda
}  // namespace chainerx
