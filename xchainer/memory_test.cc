#include "xchainer/memory.h"

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA
#include <gtest/gtest.h>

#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA

namespace xchainer {
namespace {

template <typename T>
void ExpectDataEqual(const std::shared_ptr<void>& expected, const std::shared_ptr<void>& actual, size_t size) {
    auto expected_raw_ptr = static_cast<const T*>(expected.get());
    auto actual_raw_ptr = static_cast<const T*>(actual.get());
    for (size_t i = 0; i < size; i++) {
        EXPECT_EQ(expected_raw_ptr[i], actual_raw_ptr[i]);
    }
}

#ifdef XCHAINER_ENABLE_CUDA

TEST(MemoryTest, IsPointerCudaMemory) {
    size_t size = 3;
    {
        std::shared_ptr<void> cpu_ptr = std::make_unique<uint8_t[]>(size);
        EXPECT_FALSE(IsPointerCudaMemory(cpu_ptr.get()));
    }
    {
        void* raw_ptr = nullptr;
        cuda::CheckError(cudaMallocManaged(&raw_ptr, size, cudaMemAttachGlobal));
        auto cuda_ptr = std::shared_ptr<void>{raw_ptr, cudaFree};
        EXPECT_TRUE(IsPointerCudaMemory(cuda_ptr.get()));
    }
}

TEST(MemoryTest, Allocate) {
    size_t size = 3;
    {
        std::shared_ptr<void> ptr = Allocate(MakeDevice("cpu"), size);
        EXPECT_FALSE(IsPointerCudaMemory(ptr.get()));
    }
    {
        std::shared_ptr<void> ptr = Allocate(MakeDevice("cuda"), size);
        EXPECT_TRUE(IsPointerCudaMemory(ptr.get()));
    }
}

TEST(MemoryTest, MemoryCopy) {
    size_t size = 3;
    auto data = std::make_unique<uint8_t[]>(size);
    data[0] = 1;
    data[1] = 2;
    data[2] = 3;
    std::shared_ptr<void> cpu_src = std::move(data);
    {
        // cpu to cpu
        std::shared_ptr<void> cpu_dst = std::make_unique<uint8_t[]>(size);
        MemoryCopy(cpu_dst.get(), cpu_src.get(), size);
        ExpectDataEqual<uint8_t>(cpu_src, cpu_dst, size);
    }
    {
        // cpu to gpu
        std::shared_ptr<void> gpu_dst = Allocate(MakeDevice("cuda"), size);
        MemoryCopy(gpu_dst.get(), cpu_src.get(), size);
        ExpectDataEqual<uint8_t>(cpu_src, gpu_dst, size);
    }

    std::shared_ptr<void> gpu_src = Allocate(MakeDevice("cuda"), size);
    MemoryCopy(gpu_src.get(), cpu_src.get(), size);
    {
        // gpu to cpu
        std::shared_ptr<void> cpu_dst = std::make_unique<uint8_t[]>(size);
        MemoryCopy(cpu_dst.get(), gpu_src.get(), size);
        ExpectDataEqual<uint8_t>(gpu_src, cpu_dst, size);
    }
    {
        // gpu to gpu
        std::shared_ptr<void> gpu_dst = Allocate(MakeDevice("cuda"), size);
        MemoryCopy(gpu_dst.get(), gpu_src.get(), size);
        ExpectDataEqual<uint8_t>(gpu_src, gpu_dst, size);
    }
}

TEST(MemoryTest, MemoryFromBuffer) {
    size_t size = 3;
    auto data = std::make_unique<uint8_t[]>(size);
    data[0] = 1;
    data[1] = 2;
    data[2] = 3;
    std::shared_ptr<void> cpu_src = std::move(data);
    std::shared_ptr<void> gpu_src = Allocate(MakeDevice("cuda"), size);
    MemoryCopy(gpu_src.get(), cpu_src.get(), size);
    {
        // cpu to cpu
        std::shared_ptr<void> cpu_dst = MemoryFromBuffer(MakeDevice("cpu"), cpu_src, size);
        ExpectDataEqual<uint8_t>(cpu_src, cpu_dst, size);
        EXPECT_EQ(cpu_src.get(), cpu_dst.get());
    }
    {
        // cpu to gpu
        std::shared_ptr<void> gpu_dst = MemoryFromBuffer(MakeDevice("cuda"), cpu_src, size);
        ExpectDataEqual<uint8_t>(cpu_src, gpu_dst, size);
        EXPECT_NE(cpu_src.get(), gpu_dst.get());
    }
    {
        // gpu to cpu
        std::shared_ptr<void> cpu_dst = MemoryFromBuffer(MakeDevice("cpu"), gpu_src, size);
        ExpectDataEqual<uint8_t>(gpu_src, cpu_dst, size);
        EXPECT_NE(gpu_src.get(), cpu_dst.get());
    }
    {
        // gpu to gpu
        std::shared_ptr<void> gpu_dst = MemoryFromBuffer(MakeDevice("cuda"), gpu_src, size);
        ExpectDataEqual<uint8_t>(gpu_src, gpu_dst, size);
        EXPECT_EQ(gpu_src.get(), gpu_dst.get());
    }
}

#endif  // XCHAINER_ENABLE_CUDA

}  // namespace
}  // namespace xchainer
