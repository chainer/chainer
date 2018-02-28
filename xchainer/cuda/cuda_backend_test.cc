#include "xchainer/cuda/cuda_backend.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "xchainer/array.h"
#include "xchainer/context.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/device.h"
#include "xchainer/memory.h"
#include "xchainer/native_backend.h"

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

void ExpectArraysEqual(const Array& expected, const Array& actual) {
    EXPECT_EQ(expected.dtype(), actual.dtype());
    EXPECT_EQ(expected.shape(), actual.shape());
    VisitDtype(expected.dtype(), [&expected, &actual](auto pt) {
        using T = typename decltype(pt)::type;
        int64_t total_size = expected.GetTotalSize();
        const T* data1 = static_cast<const T*>(expected.data().get());
        const T* data2 = static_cast<const T*>(actual.data().get());
        for (int64_t i = 0; i < total_size; ++i) {
            EXPECT_EQ(data1[i], data2[i]);
        }
    });
}

TEST(CudaBackendTest, GetDeviceCount) {
    Context ctx;
    int count = 0;
    CheckError(cudaGetDeviceCount(&count));
    EXPECT_EQ(count, CudaBackend(ctx).GetDeviceCount());
}

TEST(CudaBackendTest, GetDevice) {
    Context ctx;
    CudaBackend backend{ctx};
    {
        Device& device = backend.GetDevice(0);
        EXPECT_EQ(&backend, &device.backend());
        EXPECT_EQ(0, device.index());
    }
    {
        Device& device3 = backend.GetDevice(3);
        Device& device2 = backend.GetDevice(2);
        EXPECT_EQ(&backend, &device3.backend());
        EXPECT_EQ(3, device3.index());
        EXPECT_EQ(&backend, &device2.backend());
        EXPECT_EQ(2, device2.index());
    }
    {
        EXPECT_THROW(backend.GetDevice(-1), std::out_of_range);
        EXPECT_THROW(backend.GetDevice(backend.GetDeviceCount() + 1), std::out_of_range);
    }
}

TEST(CudaBackendTest, GetName) {
    Context ctx;
    EXPECT_EQ("cuda", CudaBackend(ctx).GetName());
}

TEST(CudaBackendIncompatibleTransferTest, SupportsTransferDifferentContexts) {
    Context ctx0;
    Context ctx1;
    CudaBackend backend0{ctx0};
    CudaBackend backend1{ctx1};
    Device& device0 = backend0.GetDevice(0);
    Device& device1 = backend1.GetDevice(0);
    EXPECT_FALSE(backend0.SupportsTransfer(device0, device1));
}

TEST(CudaBackendIncompatibleTransferTest, SupportsTransferNativeBackends) {
    Context ctx;
    CudaBackend cuda_backend{ctx};
    NativeBackend native_backend0{ctx};
    NativeBackend native_backend1{ctx};
    Device& device0 = native_backend0.GetDevice(0);
    Device& device1 = native_backend1.GetDevice(0);
    EXPECT_FALSE(cuda_backend.SupportsTransfer(device0, device1));
}

template <int N>
class DerivedCudaBackend : public CudaBackend {
public:
    using CudaBackend::CudaBackend;
    std::string GetName() const override { return "derived" + std::to_string(N); }
};

TEST(CudaBackendIncompatibleTransferTest, SupportsTransferDifferentCudaBackends) {
    Context ctx;
    DerivedCudaBackend<0> backend0{ctx};
    DerivedCudaBackend<1> backend1{ctx};
    Device& device0 = backend0.GetDevice(0);
    Device& device1 = backend1.GetDevice(0);
    EXPECT_FALSE(backend0.SupportsTransfer(device0, device1));
}

// Data transfer test
class CudaBackendTransferTest : public ::testing::TestWithParam<::testing::tuple<std::string, std::string>> {};

INSTANTIATE_TEST_CASE_P(Devices, CudaBackendTransferTest, ::testing::Values(std::make_tuple("cuda:0", "cuda:0"),   // cuda:0 <-> cuda:0
                                                                            std::make_tuple("cuda:0", "cuda:1"),   // cuda:0 <-> cuda:1
                                                                            std::make_tuple("cuda:0", "native:0")  // cuda:0 <-> native:0
                                                                            ));

TEST_P(CudaBackendTransferTest, SupportsTransfer) {
    Context ctx;
    Backend& backend = ctx.GetBackend("cuda");
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));
    EXPECT_TRUE(backend.SupportsTransfer(device0, device1));
}

TEST_P(CudaBackendTransferTest, MemoryCopyFrom) {
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

TEST_P(CudaBackendTransferTest, MemoryCopyTo) {
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

TEST_P(CudaBackendTransferTest, TransferDataFrom) {
    Context ctx;
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));

    size_t bytesize = 5;
    auto data = device1.Allocate(bytesize);

    // Transfer
    // TODO(niboshi): Offset is fixed to 0
    std::tuple<std::shared_ptr<void>, size_t> tuple = device0.TransferDataFrom(device1, data, 0, bytesize);

    EXPECT_EQ(0, std::memcmp(data.get(), std::get<0>(tuple).get(), bytesize));
    // TODO(niboshi): Test offset

    // Destination is ALWAYS CUDA device
    cudaPointerAttributes attr = {};
    CheckError(cudaPointerGetAttributes(&attr, std::get<0>(tuple).get()));
    EXPECT_TRUE(attr.isManaged);
    EXPECT_EQ(device0.index(), attr.device);
}

TEST_P(CudaBackendTransferTest, TransferDataTo) {
    Context ctx;
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));

    size_t bytesize = 5;
    auto data = device0.Allocate(bytesize);

    // Transfer
    // TODO(niboshi): Offset is fixed to 0
    std::tuple<std::shared_ptr<void>, size_t> tuple = device0.TransferDataTo(device1, data, 0, bytesize);

    EXPECT_EQ(0, std::memcmp(data.get(), std::get<0>(tuple).get(), bytesize));
    // TODO(niboshi): Test offset

    if (nullptr != dynamic_cast<CudaBackend*>(&device1.backend())) {
        // Destination is CUDA device
        cudaPointerAttributes attr = {};
        CheckError(cudaPointerGetAttributes(&attr, std::get<0>(tuple).get()));
        EXPECT_TRUE(attr.isManaged);
        EXPECT_EQ(device1.index(), attr.device);
    } else {
        // Destination is native device
        EXPECT_FALSE(internal::IsPointerCudaMemory(std::get<0>(tuple).get()));
    }
}

TEST_P(CudaBackendTransferTest, ArrayToDeviceFrom) {
    Context ctx;
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));

    // Allocate the source array
    float data[] = {1.0f, 2.0f};
    auto nop = [](void* p) {
        (void)p;  // unused
    };
    Array a = Array::FromBuffer({2, 1}, Dtype::kFloat32, std::shared_ptr<float>(data, nop), device1);

    // Transfer
    Array b = a.ToDevice(device0);

    EXPECT_EQ(&b.device(), &device0);
    EXPECT_EQ(&a.device(), &device1);
    if (&device0 == &device1) {
        EXPECT_EQ(a.data().get(), b.data().get()) << "Array::ToDevice() must return alias when transferring to the same native device.";
    } else {
        EXPECT_NE(a.data().get(), b.data().get())
            << "Array::ToDevice() must not return alias when transferring to different native device.";
    }
    ExpectArraysEqual(a, b);
}

TEST_P(CudaBackendTransferTest, ArrayToDeviceTo) {
    Context ctx;
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));

    // Allocate the source array
    float data[] = {1.0f, 2.0f};
    auto nop = [](void* p) {
        (void)p;  // unused
    };
    Array a = Array::FromBuffer({2, 1}, Dtype::kFloat32, std::shared_ptr<float>(data, nop), device0);

    // Transfer
    Array b = a.ToDevice(device1);

    EXPECT_EQ(&b.device(), &device1);
    EXPECT_EQ(&a.device(), &device0);
    if (&device0 == &device1) {
        EXPECT_EQ(a.data().get(), b.data().get()) << "Array::ToDevice() must return alias when transferring to the same CUDA device.";
    } else {
        EXPECT_NE(a.data().get(), b.data().get()) << "Array::ToDevice() must not return alias when transferring to a different device.";
    }
    ExpectArraysEqual(a, b);
}

}  // namespace
}  // namespace cuda
}  // namespace xchainer
