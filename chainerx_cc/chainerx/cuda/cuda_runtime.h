#pragma once

#include <cuda_runtime.h>

#include "chainerx/error.h"

namespace chainerx {
namespace cuda {

class RuntimeError : public ChainerxError {
public:
    explicit RuntimeError(cudaError_t error);
    cudaError_t error() const noexcept { return error_; }

private:
    cudaError_t error_;
};

void CheckCudaError(cudaError_t error);

void Throw(cudaError_t error);

bool IsPointerCudaMemory(const void* ptr);

bool IsPointerManagedMemory(const void* ptr);

// Occupancy
#ifdef __CUDACC__

struct GridBlockSize {
    int grid_size;
    int block_size;
};

template <typename T>
GridBlockSize CudaOccupancyMaxPotentialBlockSize(const T& func, size_t dynamic_smem_size = 0, int block_size_limit = 0) {
    GridBlockSize ret = {};
    CheckCudaError(cudaOccupancyMaxPotentialBlockSize(&ret.grid_size, &ret.block_size, func, dynamic_smem_size, block_size_limit));
    return ret;
}

#endif  // __CUDACC__

}  // namespace cuda
}  // namespace chainerx
