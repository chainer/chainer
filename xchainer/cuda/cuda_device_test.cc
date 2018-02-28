#include "xchainer/cuda/cuda_device.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "xchainer/context.h"
#include "xchainer/cuda/cuda_backend.h"
#include "xchainer/cuda/cuda_runtime.h"

namespace xchainer {
namespace cuda {
namespace {

template <typename T>
void ExpectDataEqual(const std::shared_ptr<void>& expected, const std::shared_ptr<void>& actual, size_t size) {
    auto expected_raw_ptr = static_cast<const T*>(expected.get());
    auto actual_raw_ptr = static_cast<const T*>(actual.get());
    for (size_t i = 0; i < size; i++) {
        EXPECT_EQ(expected_raw_ptr[i], actual_raw_ptr[i]);
    }
}

TEST(CudaDeviceTest, Ctor) {
    Context ctx;
    CudaBackend backend{ctx};
    {
        CudaDevice device{backend, 0};
        EXPECT_EQ(&backend, &device.backend());
        EXPECT_EQ(0, device.index());
    }
    {
        CudaDevice device{backend, 1};
        EXPECT_EQ(&backend, &device.backend());
        EXPECT_EQ(1, device.index());
    }
}

TEST(CudaDeviceTest, Allocate) {
    Context ctx;
    CudaBackend backend{ctx};
    CudaDevice device{backend, 0};

    {
        size_t bytesize = 3;
        std::shared_ptr<void> ptr = device.Allocate(bytesize);

        cudaPointerAttributes attr = {};
        CheckError(cudaPointerGetAttributes(&attr, ptr.get()));
        EXPECT_TRUE(attr.isManaged);
        EXPECT_EQ(device.index(), attr.device);
    }
    {
        size_t bytesize = 0;
        EXPECT_NO_THROW(device.Allocate(bytesize));
    }
}

TEST(CudaDeviceTest, FromBuffer) {
    size_t size = 3;
    size_t bytesize = size * sizeof(float);
    float raw_data[] = {0, 1, 2};
    std::shared_ptr<void> src(raw_data, [](float* ptr) {
        (void)ptr;  // unused
    });

    Context ctx;
    CudaBackend backend{ctx};
    CudaDevice device{backend, 0};

    std::shared_ptr<void> dst = device.FromBuffer(src, bytesize);
    ExpectDataEqual<float>(src, dst, size);
    EXPECT_NE(src.get(), dst.get());

    cudaPointerAttributes attr = {};
    CheckError(cudaPointerGetAttributes(&attr, dst.get()));
    EXPECT_TRUE(attr.isManaged);
    EXPECT_EQ(device.index(), attr.device);
}

// TODO(sonots): Any ways to test cudaDeviceSynchronize()?
TEST(CudaDeviceTest, Synchronize) {
    Context ctx;
    CudaBackend backend{ctx};
    CudaDevice device{backend, 0};
    EXPECT_NO_THROW(device.Synchronize());
}

class CudaDeviceMemoryCopyTest : public ::testing::TestWithParam<::testing::tuple<std::string, std::string>> {};

INSTANTIATE_TEST_CASE_P(Devices, CudaDeviceMemoryCopyTest, ::testing::Values(std::make_tuple("cuda:0", "cuda:0"),   // cuda:0 <-> cuda:0
                                                                             std::make_tuple("cuda:0", "cuda:1"),   // cuda:0 <-> cuda:1
                                                                             std::make_tuple("cuda:0", "native:0")  // cuda:0 <-> native:0
                                                                             ));

TEST_P(CudaDeviceMemoryCopyTest, MemoryCopyFrom) {
    size_t size = 3;
    size_t bytesize = size * sizeof(float);
    float raw_data[] = {0, 1, 2};
    std::shared_ptr<void> src_orig(raw_data, [](float* ptr) {
        (void)ptr;  // unused
    });

    Context ctx;
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));

    std::shared_ptr<void> src = device1.FromBuffer(src_orig, bytesize);
    std::shared_ptr<void> dst = device0.Allocate(bytesize);
    device0.MemoryCopyFrom(dst.get(), src.get(), bytesize, device1);
    ExpectDataEqual<float>(src, dst, size);
}

TEST_P(CudaDeviceMemoryCopyTest, MemoryCopyTo) {
    size_t size = 3;
    size_t bytesize = size * sizeof(float);
    float raw_data[] = {0, 1, 2};
    std::shared_ptr<void> src_orig(raw_data, [](float* ptr) {
        (void)ptr;  // unused
    });

    Context ctx;
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));

    std::shared_ptr<void> src = device0.FromBuffer(src_orig, bytesize);
    std::shared_ptr<void> dst = device1.Allocate(bytesize);
    device0.MemoryCopyTo(dst.get(), src.get(), bytesize, device1);
    ExpectDataEqual<float>(src, dst, size);
}

}  // namespace
}  // namespace cuda
}  // namespace xchainer
