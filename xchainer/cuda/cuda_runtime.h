#pragma once

#include <cuda_runtime.h>
#include "xchainer/error.h"

namespace xchainer {
namespace cuda {

class RuntimeError : public XchainerError {
public:
    explicit RuntimeError(cudaError_t error);
    cudaError_t error() const noexcept { return error_; }

private:
    cudaError_t error_;
};

void CheckError(cudaError_t error);

// Device management
void CudaDeviceSynchronize();

// Stream management

// Memory management

}  // namespace cuda
}  // namespace xchainer
