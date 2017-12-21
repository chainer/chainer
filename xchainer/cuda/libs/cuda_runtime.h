#pragma once

#include <cuda_runtime.h>
#include "xchainer/error.h"

namespace xchainer {
namespace cuda {

class CudaRuntimeError : public XchainerError {
public:
    explicit CudaRuntimeError(cudaError_t error);
    cudaError_t error() const noexcept { return error_; }

private:
    cudaError_t error_;
};

void CudaCheckError(cudaError_t error);

// Device management
void CudaDeviceSynchronize();

// Stream management

// Memory management

}  // namespace cuda
}  // namespace xchainer
