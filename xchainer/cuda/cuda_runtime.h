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

// Occupancy
#ifdef __CUDACC__

struct GridBlockSize {
    int grid_size;
    int block_size;
};

template <typename T>
GridBlockSize CudaOccupancyMaxPotentialBlockSize(const T& func, size_t dynamic_smem_size = 0, int block_size_limit = 0) {
    GridBlockSize ret = {};
    CheckError(
        cudaOccupancyMaxPotentialBlockSize(&ret.grid_size, &ret.block_size, func, dynamic_smem_size, block_size_limit));
    return ret;
}

#endif  // __CUDACC__

}  // namespace cuda
}  // namespace xchainer
