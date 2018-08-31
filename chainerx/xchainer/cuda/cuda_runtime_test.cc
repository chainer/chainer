#include "chainerx/cuda/cuda_runtime.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <memory>

namespace chainerx {
namespace cuda {
namespace {

TEST(CudaRuntimeTest, IsPointerCudaMemory) {
    size_t size = 3;
    {
        std::shared_ptr<void> cpu_ptr = std::make_unique<uint8_t[]>(size);
        EXPECT_FALSE(IsPointerCudaMemory(cpu_ptr.get()));
    }
    {
        void* raw_ptr = nullptr;
        cuda::CheckCudaError(cudaMallocManaged(&raw_ptr, size, cudaMemAttachGlobal));
        auto cuda_ptr = std::shared_ptr<void>{raw_ptr, cudaFree};
        EXPECT_TRUE(IsPointerCudaMemory(cuda_ptr.get()));
    }
    {
        void* raw_ptr = nullptr;
        cuda::CheckCudaError(cudaMalloc(&raw_ptr, size));
        auto cuda_ptr = std::shared_ptr<void>{raw_ptr, cudaFree};
        EXPECT_THROW(IsPointerCudaMemory(cuda_ptr.get()), XchainerError)
                << "IsPointerCudaMemory must throw an exception if non-managed CUDA memory is given";
    }
}

}  // namespace
}  // namespace cuda
}  // namespace chainerx
