#include "chainerx/cuda/cuda_runtime.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <memory>

namespace chainerx {
namespace cuda {
namespace {

TEST(CudaRuntimeTest, IsPointerCudaMemory_CpuMemory) {
    std::shared_ptr<void> cpu_ptr = std::make_unique<uint8_t[]>(3);
    EXPECT_FALSE(IsPointerCudaMemory(cpu_ptr.get()));
}

TEST(CudaRuntimeTest, IsPointerCudaMemory_CudaManagedMemory) {
    void* raw_ptr = nullptr;
    cuda::CheckCudaError(cudaMallocManaged(&raw_ptr, 3, cudaMemAttachGlobal));
    auto cuda_ptr = std::shared_ptr<void>{raw_ptr, cudaFree};
    EXPECT_TRUE(IsPointerCudaMemory(cuda_ptr.get()));
}

TEST(CudaRuntimeTest, IsPointerCudaMemory_CudaUnmanagedMemory) {
    void* raw_ptr = nullptr;
    cuda::CheckCudaError(cudaMalloc(&raw_ptr, 3));
    auto cuda_ptr = std::shared_ptr<void>{raw_ptr, cudaFree};
    EXPECT_TRUE(IsPointerCudaMemory(cuda_ptr.get()));
}

}  // namespace
}  // namespace cuda
}  // namespace chainerx
