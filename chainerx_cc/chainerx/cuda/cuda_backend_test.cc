#include "chainerx/cuda/cuda_backend.h"

#include <tuple>
#include <utility>
#include <vector>

#include <absl/types/optional.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "chainerx/array.h"
#include "chainerx/context.h"
#include "chainerx/cuda/cuda_runtime.h"
#include "chainerx/device.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/native/native_backend.h"
#include "chainerx/routines/creation.h"
#include "chainerx/testing/threading.h"
#include "chainerx/testing/util.h"
#include "chainerx/util.h"

namespace chainerx {
namespace cuda {
namespace {

template <typename T>
std::shared_ptr<void> ToHost(const Device& from_device, const std::shared_ptr<void>& mem, size_t size) {
    if (nullptr != dynamic_cast<native::NativeBackend*>(&from_device.backend())) {
        return mem;
    }
    std::shared_ptr<void> host_mem = std::make_unique<T[]>(size);
    cudaMemcpy(host_mem.get(), mem.get(), sizeof(T) * size, cudaMemcpyDeviceToHost);
    return host_mem;
}

template <typename T>
void ExpectDataEqual(
        const std::shared_ptr<void>& expected,
        const std::shared_ptr<void>& actual,
        size_t size,
        Device& expected_ptr_device,
        Device& actual_ptr_device) {
    std::shared_ptr<void> expected_host = ToHost<T>(expected_ptr_device, expected, size);
    std::shared_ptr<void> actual_host = ToHost<T>(actual_ptr_device, actual, size);

    auto expected_raw_ptr = static_cast<const T*>(expected_host.get());
    auto actual_raw_ptr = static_cast<const T*>(actual_host.get());

    expected_ptr_device.Synchronize();
    actual_ptr_device.Synchronize();

    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(expected_raw_ptr[i], actual_raw_ptr[i]);
    }
}

void ExpectArraysEqual(const Array& expected, const Array& actual) {
    EXPECT_EQ(expected.dtype(), actual.dtype());
    EXPECT_EQ(expected.shape(), actual.shape());

    VisitDtype(expected.dtype(), [&expected, &actual](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> expected_iarray{expected};
        IndexableArray<const T> actual_iarray{actual};
        Indexer<> indexer{expected.shape()};

        actual.device().Synchronize();
        expected.device().Synchronize();

        ExpectDataEqual<T>(expected.data(), actual.data(), expected.GetTotalSize(), expected.device(), actual.device());
    });
}

TEST(CudaBackendTest, GetDeviceCount) {
    Context ctx;
    int count = 0;
    CheckCudaError(cudaGetDeviceCount(&count));
    EXPECT_EQ(count, CudaBackend(ctx).GetDeviceCount());
}

TEST(NativeBackendTest, IsNative) {
    Context ctx;
    EXPECT_FALSE(CudaBackend{ctx}.IsNative());
}

TEST(CudaBackendTest, GetDeviceCountGetNameThreadSafe) {
    Context ctx;
    CudaBackend backend{ctx};
    int expected_device_count = backend.GetDeviceCount();
    std::string expected_backend_name = backend.GetName();

    testing::RunThreads(2, [&backend, expected_device_count, &expected_backend_name]() {
        int device_count = backend.GetDeviceCount();
        std::string name = backend.GetName();
        EXPECT_EQ(expected_device_count, device_count);
        EXPECT_EQ(expected_backend_name, name);
    });
}

TEST(CudaBackendTest, GetDevice) {
    Context ctx;
    CudaBackend backend{ctx};

    Device& device = backend.GetDevice(0);
    EXPECT_EQ(&backend, &device.backend());
    EXPECT_EQ(0, device.index());
}

TEST(CudaBackendTest, GetDeviceThreadSafe) {
    Context ctx;
    CudaBackend backend{ctx};

    testing::RunThreads(2, [&backend]() {
        Device& device = backend.GetDevice(0);
        EXPECT_EQ(&backend, &device.backend());
        EXPECT_EQ(0, device.index());
    });
}

TEST(CudaBackendTest, GetDeviceSecondDevice) {
    CHAINERX_REQUIRE_DEVICE("cuda", 2);
    Context ctx;
    CudaBackend backend{ctx};

    Device& device1 = backend.GetDevice(1);
    EXPECT_EQ(&backend, &device1.backend());
    EXPECT_EQ(1, device1.index());
}

TEST(CudaBackendTest, GetDeviceOutOfRange) {
    Context ctx;
    CudaBackend backend{ctx};

    EXPECT_THROW(backend.GetDevice(-1), std::out_of_range);
    EXPECT_THROW(backend.GetDevice(backend.GetDeviceCount() + 1), std::out_of_range);
}

TEST(CudaBackendTest, GetName) {
    Context ctx;
    EXPECT_EQ("cuda", CudaBackend(ctx).GetName());
}

TEST(CudaBackendTest, SupportsTransferThreadSafe) {
    CHAINERX_REQUIRE_DEVICE("cuda", 2);
    static constexpr size_t kThreadCount = 2;

    Context ctx0{};
    Context ctx1{};
    Backend& ctx0_backend = ctx0.GetBackend("cuda");
    Backend& ctx1_backend = ctx1.GetBackend("cuda");
    Device& ctx0_device0 = ctx0_backend.GetDevice(0);
    Device& ctx0_device1 = ctx0_backend.GetDevice(1);
    Device& ctx1_device = ctx1_backend.GetDevice(0);

    testing::RunThreads(kThreadCount, [&ctx0_backend, &ctx0_device0, &ctx0_device1, &ctx1_device]() {
        EXPECT_TRUE(ctx0_backend.SupportsTransfer(ctx0_device0, ctx0_device1));
        EXPECT_FALSE(ctx0_backend.SupportsTransfer(ctx0_device0, ctx1_device));
    });
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
    native::NativeBackend native_backend0{ctx};
    native::NativeBackend native_backend1{ctx};
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
class CudaBackendTransferTest : public ::testing::TestWithParam<::testing::tuple<std::string, std::string, int>> {};

INSTANTIATE_TEST_CASE_P(
        Devices,
        CudaBackendTransferTest,
        ::testing::Values(
                // 3rd parameter is the number of required CUDA devices to run the test.
                std::make_tuple("cuda:0", "cuda:0", 1),  // cuda:0 <-> cuda:0
                std::make_tuple("cuda:0", "cuda:1", 2),  // cuda:0 <-> cuda:1
                std::make_tuple("cuda:0", "native:0", 1)));  // cuda:0 <-> native:0

TEST_P(CudaBackendTransferTest, SupportsTransfer) {
    CHAINERX_REQUIRE_DEVICE("cuda", ::testing::get<2>(GetParam()));
    Context ctx;
    Backend& backend = ctx.GetBackend("cuda");
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));
    EXPECT_TRUE(backend.SupportsTransfer(device0, device1));
}

TEST_P(CudaBackendTransferTest, MemoryCopyFrom) {
    CHAINERX_REQUIRE_DEVICE("cuda", ::testing::get<2>(GetParam()));
    size_t size = 3;
    size_t bytesize = size * sizeof(float);
    float raw_data[] = {0, 1, 2};
    std::shared_ptr<void> src_orig(raw_data, [](float*) {});

    Context ctx;
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));

    std::shared_ptr<void> src = device1.FromHostMemory(src_orig, bytesize);
    std::shared_ptr<void> dst = device0.Allocate(bytesize);
    device0.MemoryCopyFrom(dst.get(), src.get(), bytesize, device1);

    ExpectDataEqual<float>(src, dst, size, device1, device0);
}

TEST_P(CudaBackendTransferTest, MemoryCopyFromZeroByte) {
    CHAINERX_REQUIRE_DEVICE("cuda", ::testing::get<2>(GetParam()));
    size_t size = 0;
    size_t bytesize = 0;
    std::shared_ptr<void> src_orig(nullptr, [](float*) {});

    Context ctx;
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));

    std::shared_ptr<void> src = device1.FromHostMemory(src_orig, bytesize);
    std::shared_ptr<void> dst = device0.Allocate(bytesize);
    device0.MemoryCopyFrom(dst.get(), src.get(), bytesize, device1);

    ExpectDataEqual<float>(src, dst, size, device1, device0);
}

TEST_P(CudaBackendTransferTest, MemoryCopyTo) {
    CHAINERX_REQUIRE_DEVICE("cuda", ::testing::get<2>(GetParam()));
    size_t size = 3;
    size_t bytesize = size * sizeof(float);
    float raw_data[] = {0, 1, 2};
    std::shared_ptr<void> src_orig(raw_data, [](float*) {});

    Context ctx;
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));

    std::shared_ptr<void> src = device0.FromHostMemory(src_orig, bytesize);
    std::shared_ptr<void> dst = device1.Allocate(bytesize);
    device0.MemoryCopyTo(dst.get(), src.get(), bytesize, device1);

    ExpectDataEqual<float>(src, dst, size, device0, device1);
}

TEST_P(CudaBackendTransferTest, MemoryCopyToZeroByte) {
    CHAINERX_REQUIRE_DEVICE("cuda", ::testing::get<2>(GetParam()));
    size_t size = 0;
    size_t bytesize = 0;
    std::shared_ptr<void> src_orig(nullptr, [](float*) {});

    Context ctx;
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));

    std::shared_ptr<void> src = device0.FromHostMemory(src_orig, bytesize);
    std::shared_ptr<void> dst = device1.Allocate(bytesize);
    device0.MemoryCopyTo(dst.get(), src.get(), bytesize, device1);

    ExpectDataEqual<float>(src, dst, size, device0, device1);
}

TEST_P(CudaBackendTransferTest, TransferDataFrom) {
    CHAINERX_REQUIRE_DEVICE("cuda", ::testing::get<2>(GetParam()));
    Context ctx;
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));

    size_t bytesize = 5;
    auto data = device1.Allocate(bytesize);

    // Transfer
    std::shared_ptr<void> trans_data = device0.TransferDataFrom(device1, data, 0, bytesize);
    device0.Synchronize();

    ExpectDataEqual<uint8_t>(data, trans_data, bytesize, device1, device0);

    // Destination is ALWAYS CUDA device
    cudaPointerAttributes attr = {};
    CheckCudaError(cudaPointerGetAttributes(&attr, trans_data.get()));
    EXPECT_EQ(device0.index(), attr.device);
    EXPECT_FALSE(IsPointerManagedMemory(trans_data.get()));
}

TEST_P(CudaBackendTransferTest, TransferDataTo) {
    CHAINERX_REQUIRE_DEVICE("cuda", ::testing::get<2>(GetParam()));
    Context ctx;
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));

    size_t bytesize = 5;
    auto data = device0.Allocate(bytesize);

    // Transfer
    std::shared_ptr<void> trans_data = device0.TransferDataTo(device1, data, 0, bytesize);
    device0.Synchronize();

    ExpectDataEqual<uint8_t>(data, trans_data, bytesize, device0, device1);

    if (nullptr != dynamic_cast<CudaBackend*>(&device1.backend())) {
        // Destination is CUDA device
        cudaPointerAttributes attr = {};
        CheckCudaError(cudaPointerGetAttributes(&attr, trans_data.get()));
        EXPECT_EQ(device1.index(), attr.device);
        EXPECT_FALSE(IsPointerManagedMemory(trans_data.get()));
    } else {
        // Destination is native device
        EXPECT_FALSE(IsPointerCudaMemory(trans_data.get()));
    }
}

TEST_P(CudaBackendTransferTest, ArrayToDeviceFrom) {
    CHAINERX_REQUIRE_DEVICE("cuda", ::testing::get<2>(GetParam()));
    Context ctx;
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));

    // Allocate the source array
    float data[] = {1.0f, 2.0f};
    auto nop = [](void* p) {
        (void)p;  // unused
    };
    Array a = FromContiguousHostData({2, 1}, Dtype::kFloat32, std::shared_ptr<float>(data, nop), device1);

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
    CHAINERX_REQUIRE_DEVICE("cuda", ::testing::get<2>(GetParam()));
    Context ctx;
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));

    // Allocate the source array
    float data[] = {1.0f, 2.0f};
    auto nop = [](void* p) {
        (void)p;  // unused
    };
    Array a = FromContiguousHostData({2, 1}, Dtype::kFloat32, std::shared_ptr<float>(data, nop), device0);

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

class EnvVarScope {
public:
    EnvVarScope(std::string name, const std::string& value) : name_(std::move(name)), old_value_{GetEnv(name_)} { SetEnv(name_, value); }

    ~EnvVarScope() {
        if (old_value_) {
            SetEnv(name_, *old_value_);
        } else {
            UnsetEnv(name_);
        }
    }

private:
    const std::string name_{};
    absl::optional<std::string> old_value_{};
};

TEST(CudaBackendTest, GetCudnnMaxWorkspaceSize) {
    Context ctx;
    {
        CudaBackend backend{ctx};
        EXPECT_EQ(CudaBackend::kCudnnDefaultMaxWorkspaceSize, backend.GetCudnnMaxWorkspaceSize());
    }
    {
        CudaBackend backend{ctx};
        backend.SetCudnnMaxWorkspaceSize(10);
        EXPECT_EQ(size_t{10}, backend.GetCudnnMaxWorkspaceSize());
        backend.SetCudnnMaxWorkspaceSize(0);
        EXPECT_EQ(size_t{0}, backend.GetCudnnMaxWorkspaceSize());
    }
    {
        CudaBackend backend{ctx};
        {
            EnvVarScope scope{CudaBackend::kCudnnMaxWorkspaceSizeEnvVarName, "10"};
            EXPECT_EQ(size_t{10}, backend.GetCudnnMaxWorkspaceSize());
        }
        {
            // env is cached on the first access, so not reflected.
            EnvVarScope scope{CudaBackend::kCudnnMaxWorkspaceSizeEnvVarName, "0"};
            EXPECT_EQ(size_t{10}, backend.GetCudnnMaxWorkspaceSize());
        }
    }
}

TEST(CudaBackendTest, SetAndGetCudnnMaxWorkspaceSizeThreadSafe) {
    Context ctx;
    CudaBackend backend{ctx};

    testing::RunThreads(2, [&backend]() {
        backend.SetCudnnMaxWorkspaceSize(10);
        EXPECT_EQ(size_t{10}, backend.GetCudnnMaxWorkspaceSize());
    });
}

}  // namespace
}  // namespace cuda
}  // namespace chainerx
