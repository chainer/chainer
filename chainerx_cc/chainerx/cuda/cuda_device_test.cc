#include "chainerx/cuda/cuda_device.h"

#include <cstddef>
#include <memory>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "chainerx/array.h"
#include "chainerx/context.h"
#include "chainerx/cuda/cuda_backend.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/device.h"
#include "chainerx/kernels/linalg.h"
#include "chainerx/testing/array.h"
#include "chainerx/testing/array_check.h"
#include "chainerx/testing/device_session.h"
#include "chainerx/testing/threading.h"
#include "chainerx/testing/util.h"

namespace chainerx {
namespace cuda {
namespace {

std::shared_ptr<void> ToHost(const std::shared_ptr<void>& cuda_mem, size_t bytesize) {
    std::shared_ptr<void> host_mem = std::make_unique<char[]>(bytesize);
    cudaMemcpy(host_mem.get(), cuda_mem.get(), bytesize, cudaMemcpyDeviceToHost);
    return host_mem;
}

template <typename T>
void ExpectDataEqual(const std::shared_ptr<void>& expected, const std::shared_ptr<void>& actual, size_t size, Device& ptr_device) {
    auto expected_raw_ptr = static_cast<const T*>(expected.get());
    auto actual_raw_ptr = static_cast<const T*>(actual.get());

    ptr_device.Synchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(expected_raw_ptr[i], actual_raw_ptr[i]);
    }
}

CudaDevice& GetCudaDevice(Context& ctx, int device_index) {
    // Using dynamic_cast to ensure it's actually CudaDevice
    return dynamic_cast<CudaDevice&>(ctx.GetDevice({"cuda", device_index}));
}

TEST(CudaDeviceTest, Allocate) {
    Context ctx;
    CudaDevice& device = GetCudaDevice(ctx, 0);

    size_t bytesize = 3;
    std::shared_ptr<void> ptr = device.Allocate(bytesize);
    EXPECT_NE(ptr, nullptr);

    cudaPointerAttributes attr = {};
    CheckCudaError(cudaPointerGetAttributes(&attr, ptr.get()));
    EXPECT_EQ(device.index(), attr.device);
    EXPECT_FALSE(IsPointerManagedMemory(ptr.get()));
}

TEST(CudaDeviceTest, AllocateZero) {
    Context ctx;
    CudaDevice& device = GetCudaDevice(ctx, 0);

    std::shared_ptr<void> ptr = device.Allocate(size_t{0});
    EXPECT_EQ(ptr, nullptr);
}

TEST(CudaDeviceTest, AllocateFreeThreadSafe) {
    static constexpr size_t kNumThreads = 2;
    static constexpr size_t kNumLoopsPerThread = 1;
    Context ctx;
    CudaDevice& device = GetCudaDevice(ctx, 0);

    // Allocate-and-free loop
    auto func = [&device](size_t size) {
        for (size_t j = 0; j < kNumLoopsPerThread; ++j) {
            std::shared_ptr<void> ptr = device.Allocate(size);
            (void)ptr;  // unused
        }
    };

    // Launch threads
    std::vector<std::thread> threads;
    threads.reserve(kNumThreads);
    for (size_t i = 0; i < kNumThreads; ++i) {
        threads.emplace_back(func, i);
    }

    // Join threads
    for (size_t i = 0; i < kNumThreads; ++i) {
        threads[i].join();
    }
}

TEST(CudaDeviceTest, MakeDataFromForeignPointer) {
    Context ctx;
    CudaDevice& device = GetCudaDevice(ctx, 0);

    std::shared_ptr<void> cuda_data = device.Allocate(3);
    EXPECT_EQ(cuda_data.get(), device.MakeDataFromForeignPointer(cuda_data).get());
}

TEST(CudaDeviceTest, MakeDataFromForeignPointer_NonCudaMemory) {
    Context ctx;
    CudaDevice& device = GetCudaDevice(ctx, 0);

    std::shared_ptr<void> cpu_data = std::make_unique<uint8_t[]>(3);
    EXPECT_THROW(device.MakeDataFromForeignPointer(cpu_data), ChainerxError) << "must throw an exception if non CUDA memory is given";
}

TEST(CudaDeviceTest, MakeDataFromForeignPointer_NonUnifiedMemory) {
    Context ctx;
    CudaDevice& device = GetCudaDevice(ctx, 0);

    void* raw_ptr = nullptr;
    cuda::CheckCudaError(cudaMalloc(&raw_ptr, 3));
    auto cuda_data = std::shared_ptr<void>{raw_ptr, cudaFree};
    EXPECT_EQ(cuda_data.get(), device.MakeDataFromForeignPointer(cuda_data).get());
}

TEST(CudaDeviceTest, MakeDataFromForeignPointer_FromAnotherDevice) {
    CHAINERX_REQUIRE_DEVICE("cuda", 2);
    Context ctx;

    CudaDevice& device = GetCudaDevice(ctx, 0);
    CudaDevice& another_device = GetCudaDevice(ctx, 1);

    std::shared_ptr<void> cuda_data = another_device.Allocate(3);
    EXPECT_THROW(device.MakeDataFromForeignPointer(cuda_data), ChainerxError)
            << "must throw an exception if CUDA memory resides on another device";
}

TEST(CudaDeviceTest, FromHostMemory) {
    size_t size = 3;
    size_t bytesize = size * sizeof(float);
    float raw_data[] = {0, 1, 2};
    std::shared_ptr<void> src(raw_data, [](const float* ptr) {
        (void)ptr;  // unused
    });

    Context ctx;
    CudaDevice& device = GetCudaDevice(ctx, 0);

    std::shared_ptr<void> dst = device.FromHostMemory(src, bytesize);
    EXPECT_NE(src.get(), dst.get());

    ExpectDataEqual<float>(src, ToHost(dst, bytesize), size, device);

    cudaPointerAttributes attr = {};
    CheckCudaError(cudaPointerGetAttributes(&attr, dst.get()));
    EXPECT_EQ(device.index(), attr.device);
    EXPECT_FALSE(IsPointerManagedMemory(dst.get()));
}

TEST(CudaDeviceTest, DotNonContiguousOut) {
    testing::DeviceSession session{{"cuda", 0}};
    Array a = testing::BuildArray({2, 3}).WithLinearData(1.f);
    Array b = testing::BuildArray({3, 2}).WithData<float>({1.f, 2.f, -1.f, -3.f, 2.f, 4.f});
    Array c = testing::BuildArray({2, 2}).WithData<float>({0.f, 0.f, 0.f, 0.f}).WithPadding(1);
    a.device().backend().CallKernel<DotKernel>(a, b, c);

    Array e = testing::BuildArray({2, 2}).WithData<float>({5.f, 8.f, 11.f, 17.f});
    EXPECT_ARRAY_EQ(e, c);
}

// TODO(sonots): Any ways to test cudaDeviceSynchronize()?
TEST(CudaDeviceTest, Synchronize) {
    Context ctx{};
    CudaDevice& device = GetCudaDevice(ctx, 0);
    device.Synchronize();  // no throw
}

TEST(CudaDeviceTest, MemoryPoolHook) {
    Context ctx{};
    CudaDevice& device = GetCudaDevice(ctx, 0);
    const std::shared_ptr<MemoryPool> memory_pool = device.device_memory_pool();
    bool called = false;
    auto hook = [&called](MemoryPool&, size_t) { called = true; };
    memory_pool->SetMallocPreprocessHook(hook);

    EXPECT_FALSE(called);
    device.Allocate(1);
    EXPECT_TRUE(called);
}

}  // namespace
}  // namespace cuda
}  // namespace chainerx
