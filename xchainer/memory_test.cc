#include "xchainer/memory.h"

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA
#include <gtest/gtest.h>

#include "xchainer/backend.h"
#include "xchainer/context.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_backend.h"
#include "xchainer/cuda/cuda_device.h"
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/native_backend.h"
#include "xchainer/native_device.h"

namespace xchainer {
namespace internal {
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
    {
        void* raw_ptr = nullptr;
        cuda::CheckError(cudaMalloc(&raw_ptr, size));
        auto cuda_ptr = std::shared_ptr<void>{raw_ptr, cudaFree};
        EXPECT_THROW(IsPointerCudaMemory(cuda_ptr.get()), XchainerError)
            << "IsPointerCudaMemory must throw an exception if non-managed CUDA memory is given";
    }
}

TEST(MemoryTest, Allocate) {
    Context ctx;
    size_t size = 3;
    {
        NativeBackend native_backend{ctx};
        NativeDevice native_device{native_backend, 0};
        std::shared_ptr<void> ptr = Allocate(native_device, size);
        EXPECT_FALSE(IsPointerCudaMemory(ptr.get()));
    }
    {
        cuda::CudaBackend cuda_backend{ctx};
        cuda::CudaDevice cuda_device{cuda_backend, 0};
        std::shared_ptr<void> ptr = Allocate(cuda_device, size);
        EXPECT_TRUE(IsPointerCudaMemory(ptr.get()));
    }
}

TEST(MemoryTest, MemoryCopy) {
    size_t size = 3;
    size_t bytesize = size * sizeof(float);
    float raw_data[] = {0, 1, 2};
    std::shared_ptr<void> cpu_src(raw_data, [](float* ptr) {
        (void)ptr;  // unused
    });

    Context ctx;
    cuda::CudaBackend cuda_backend{ctx};
    cuda::CudaDevice cuda_device{cuda_backend, 0};

    {
        // cpu to cpu
        std::shared_ptr<void> cpu_dst = std::make_unique<float[]>(size);
        MemoryCopy(cpu_dst.get(), cpu_src.get(), bytesize);
        ExpectDataEqual<float>(cpu_src, cpu_dst, size);
    }
    {
        // cpu to gpu
        std::shared_ptr<void> gpu_dst = Allocate(cuda_device, bytesize);
        MemoryCopy(gpu_dst.get(), cpu_src.get(), bytesize);
        ExpectDataEqual<float>(cpu_src, gpu_dst, size);
    }

    std::shared_ptr<void> gpu_src = Allocate(cuda_device, bytesize);
    MemoryCopy(gpu_src.get(), cpu_src.get(), bytesize);
    {
        // gpu to cpu
        std::shared_ptr<void> cpu_dst = std::make_unique<float[]>(size);
        MemoryCopy(cpu_dst.get(), gpu_src.get(), bytesize);
        ExpectDataEqual<float>(gpu_src, cpu_dst, size);
    }
    {
        // gpu to gpu
        std::shared_ptr<void> gpu_dst = Allocate(cuda_device, bytesize);
        MemoryCopy(gpu_dst.get(), gpu_src.get(), bytesize);
        ExpectDataEqual<float>(gpu_src, gpu_dst, size);
    }
}

TEST(MemoryTest, MemoryFromBuffer) {
    size_t size = 3;
    size_t bytesize = size * sizeof(float);
    float raw_data[] = {0, 1, 2};
    std::shared_ptr<void> cpu_src(raw_data, [](float* ptr) {
        (void)ptr;  // unused
    });

    Context ctx;
    NativeBackend native_backend{ctx};
    NativeDevice native_device{native_backend, 0};
    cuda::CudaBackend cuda_backend{ctx};
    cuda::CudaDevice cuda_device{cuda_backend, 0};

    std::shared_ptr<void> gpu_src = Allocate(cuda_device, bytesize);
    MemoryCopy(gpu_src.get(), cpu_src.get(), size);
    {
        // cpu to cpu
        std::shared_ptr<void> cpu_dst = MemoryFromBuffer(native_device, cpu_src, bytesize);
        ExpectDataEqual<float>(cpu_src, cpu_dst, size);
        EXPECT_EQ(cpu_src.get(), cpu_dst.get());
    }
    {
        // cpu to gpu
        std::shared_ptr<void> gpu_dst = MemoryFromBuffer(cuda_device, cpu_src, bytesize);
        ExpectDataEqual<float>(cpu_src, gpu_dst, size);
        EXPECT_NE(cpu_src.get(), gpu_dst.get());
    }
    {
        // gpu to cpu
        std::shared_ptr<void> cpu_dst = MemoryFromBuffer(native_device, gpu_src, bytesize);
        ExpectDataEqual<float>(gpu_src, cpu_dst, size);
        EXPECT_NE(gpu_src.get(), cpu_dst.get());
    }
    {
        // gpu to gpu
        std::shared_ptr<void> gpu_dst = MemoryFromBuffer(cuda_device, gpu_src, bytesize);
        ExpectDataEqual<float>(gpu_src, gpu_dst, size);
        EXPECT_EQ(gpu_src.get(), gpu_dst.get());
    }
}

#endif  // XCHAINER_ENABLE_CUDA

}  // namespace
}  // namespace internal
}  // namespace xchainer
