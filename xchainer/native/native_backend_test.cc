#include "xchainer/native/native_backend.h"

#include <cstring>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "xchainer/array.h"
#include "xchainer/context.h"
#include "xchainer/device.h"
#include "xchainer/routines/creation.h"
#include "xchainer/testing/threading.h"

namespace xchainer {
namespace native {
namespace {

template <typename T>
void ExpectDataEqual(const std::shared_ptr<void>& expected, const std::shared_ptr<void>& actual, size_t size) {
    auto expected_raw_ptr = static_cast<const T*>(expected.get());
    auto actual_raw_ptr = static_cast<const T*>(actual.get());
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(expected_raw_ptr[i], actual_raw_ptr[i]);
    }
}

void ExpectArraysEqual(const Array& expected, const Array& actual) {
    EXPECT_EQ(expected.dtype(), actual.dtype());
    EXPECT_EQ(expected.shape(), actual.shape());
    VisitDtype(expected.dtype(), [&expected, &actual](auto pt) {
        using T = typename decltype(pt)::type;
        int64_t total_size = expected.GetTotalSize();
        auto data1 = static_cast<const T*>(expected.data().get());
        auto data2 = static_cast<const T*>(actual.data().get());
        for (int64_t i = 0; i < total_size; ++i) {
            EXPECT_EQ(data1[i], data2[i]);
        }
    });
}

TEST(NativeBackendTest, GetDeviceCount) {
    Context ctx;
    // TODO(sonots): Get number of CPU cores
    EXPECT_EQ(4, NativeBackend{ctx}.GetDeviceCount());
}

TEST(NativeBackendTest, GetDeviceCountGetNameThreadSafe) {
    static constexpr size_t kRepeat = 10;
    static constexpr size_t kThreadCount = 32;
    static constexpr size_t kRepeatCountPerThread = 100;
    int expected_device_count = Context{}.GetNativeBackend().GetDeviceCount();
    std::string expected_backend_name = Context{}.GetNativeBackend().GetName();

    xchainer::testing::CheckThreadSafety(
            kRepeat,
            kThreadCount,
            [](size_t /*repeat*/) {
                auto ctx = std::make_unique<Context>();
                Backend& backend = ctx->GetNativeBackend();
                return std::tuple<std::unique_ptr<Context>, Backend*>{std::move(ctx), &backend};
            },
            [expected_device_count, &expected_backend_name](size_t /*thread_index*/, const auto& tup) {
                Backend& backend = *std::get<1>(tup);
                for (size_t i = 0; i < kRepeatCountPerThread; ++i) {
                    int device_count = backend.GetDeviceCount();
                    std::string name = backend.GetName();
                    EXPECT_EQ(expected_device_count, device_count);
                    EXPECT_EQ(expected_backend_name, name);
                }
                return nullptr;
            },
            [](const std::vector<std::nullptr_t>& /*results*/) {});
}

TEST(NativeBackendTest, GetDevice) {
    Context ctx;
    NativeBackend backend{ctx};
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

TEST(NativeBackendTest, GetDeviceThreadSafe) {
    static constexpr size_t kRepeat = 10;
    static constexpr int kDeviceCount = 4;
    static constexpr size_t kThreadCountPerDevice = 32;
    static constexpr size_t kThreadCount = kDeviceCount * kThreadCountPerDevice;

    xchainer::testing::CheckThreadSafety(
            kRepeat,
            kThreadCount,
            [](size_t /*repeat*/) {
                auto ctx = std::make_unique<Context>();
                Backend& backend = ctx->GetNativeBackend();
                return std::tuple<std::unique_ptr<Context>, Backend*>{std::move(ctx), &backend};
            },
            [](size_t thread_index, const auto& tup) {
                Backend& backend = *std::get<1>(tup);
                int device_index = thread_index / kThreadCountPerDevice;
                Device& device = backend.GetDevice(device_index);
                EXPECT_EQ(&backend, &device.backend());
                return &device;
            },
            [](const std::vector<Device*>& results) {
                // Check device pointers are identical within each set of threads corresponding to one device
                for (int device_index = 0; device_index < kDeviceCount; ++device_index) {
                    auto it_first = std::next(results.begin(), device_index * kThreadCountPerDevice);
                    auto it_last = std::next(results.begin(), (device_index + 1) * kThreadCountPerDevice);
                    Device* ref_device = *it_first;

                    // Check the device index
                    ASSERT_EQ(device_index, ref_device->index());

                    for (auto it = it_first; it != it_last; ++it) {
                        ASSERT_EQ(ref_device, *it);
                    }
                }
            });
}

TEST(NativeBackendTest, GetName) {
    Context ctx;
    EXPECT_EQ("native", NativeBackend{ctx}.GetName());
}

TEST(NativeBackendTest, SupportsTransferThreadSafe) {
    static constexpr size_t kRepeat = 10;
    static constexpr size_t kThreadCount = 32;
    static constexpr size_t kRepeatCountPerThread = 100;

    struct CheckContext {
        std::unique_ptr<Context> context0;
        std::unique_ptr<Context> context1;
        Backend& context0_backend;
        Backend& context1_backend;
        Device& context0_device0;
        Device& context0_device1;
        Device& context1_device;
    };

    xchainer::testing::CheckThreadSafety(
            kRepeat,
            kThreadCount,
            [](size_t /*repeat*/) {
                auto ctx0 = std::make_unique<Context>();
                auto ctx1 = std::make_unique<Context>();
                Backend& context0_backend = ctx0->GetNativeBackend();
                Backend& context1_backend = ctx1->GetNativeBackend();
                return CheckContext{std::move(ctx0),
                                    std::move(ctx1),
                                    context0_backend,
                                    context1_backend,
                                    context0_backend.GetDevice(0),
                                    context0_backend.GetDevice(1),
                                    context1_backend.GetDevice(0)};
            },
            [](size_t /*thread_index*/, const CheckContext& check_ctx) {
                Backend& context0_backend = check_ctx.context0_backend;
                Device& context0_device0 = check_ctx.context0_device0;
                Device& context0_device1 = check_ctx.context0_device1;
                Device& context1_device = check_ctx.context1_device;
                for (size_t i = 0; i < kRepeatCountPerThread; ++i) {
                    EXPECT_TRUE(context0_backend.SupportsTransfer(context0_device0, context0_device1));
                    EXPECT_FALSE(context0_backend.SupportsTransfer(context0_device0, context1_device));
                }
                return nullptr;
            },
            [](const std::vector<std::nullptr_t>& /*results*/) {});
}

TEST(NativeBackendIncompatibleTransferTest, SupportsTransferDifferentContexts) {
    Context ctx0;
    Context ctx1;
    NativeBackend backend0{ctx0};
    NativeBackend backend1{ctx1};
    Device& device0 = backend0.GetDevice(0);
    Device& device1 = backend1.GetDevice(0);
    EXPECT_FALSE(backend0.SupportsTransfer(device0, device1));
}

template <int N>
class DerivedNativeBackend : public NativeBackend {
public:
    using NativeBackend::NativeBackend;
    std::string GetName() const override { return "derived" + std::to_string(N); }
};

TEST(NativeBackendIncompatibleTransferTest, SupportsTransferDifferentNativeBackends) {
    Context ctx;
    DerivedNativeBackend<0> backend0{ctx};
    DerivedNativeBackend<1> backend1{ctx};
    Device& device0 = backend0.GetDevice(0);
    Device& device1 = backend1.GetDevice(0);
    EXPECT_FALSE(backend0.SupportsTransfer(device0, device1));
}

// Data transfer test
class NativeBackendTransferTest : public ::testing::TestWithParam<::testing::tuple<std::string, std::string>> {};

INSTANTIATE_TEST_CASE_P(
        Devices,
        NativeBackendTransferTest,
        ::testing::Values(
                std::make_tuple("native:0", "native:0"),  // native:0 <-> native:0
                std::make_tuple("native:0", "native:1")));  // native:0 <-> native:1

TEST_P(NativeBackendTransferTest, SupportsTransfer) {
    Context ctx;
    Backend& backend = ctx.GetBackend("native");
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));
    EXPECT_TRUE(backend.SupportsTransfer(device0, device1));
}

TEST_P(NativeBackendTransferTest, MemoryCopyFrom) {
    size_t size = 3;
    size_t bytesize = size * sizeof(float);
    float raw_data[] = {0, 1, 2};
    std::shared_ptr<void> src_orig(raw_data, [](const float* ptr) {
        (void)ptr;  // unused
    });

    Context ctx;
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));

    std::shared_ptr<void> src = device1.FromHostMemory(src_orig, bytesize);
    std::shared_ptr<void> dst = device0.Allocate(bytesize);
    device0.MemoryCopyFrom(dst.get(), src.get(), bytesize, device1);
    ExpectDataEqual<float>(src, dst, size);
}

TEST_P(NativeBackendTransferTest, MemoryCopyTo) {
    size_t size = 3;
    size_t bytesize = size * sizeof(float);
    float raw_data[] = {0, 1, 2};
    std::shared_ptr<void> src_orig(raw_data, [](const float* ptr) {
        (void)ptr;  // unused
    });

    Context ctx;
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));

    std::shared_ptr<void> src = device0.FromHostMemory(src_orig, bytesize);
    std::shared_ptr<void> dst = device1.Allocate(bytesize);
    device0.MemoryCopyTo(dst.get(), src.get(), bytesize, device1);
    ExpectDataEqual<float>(src, dst, size);
}

TEST_P(NativeBackendTransferTest, TransferDataFrom) {
    Context ctx;
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));

    size_t bytesize = 5;
    auto data = device1.Allocate(bytesize);

    // Transfer
    std::shared_ptr<void> trans_data = device0.TransferDataFrom(device1, data, 0, bytesize);

    EXPECT_EQ(0, std::memcmp(data.get(), trans_data.get(), bytesize));
}

TEST_P(NativeBackendTransferTest, TransferDataTo) {
    Context ctx;
    Device& device0 = ctx.GetDevice(::testing::get<0>(GetParam()));
    Device& device1 = ctx.GetDevice(::testing::get<1>(GetParam()));

    size_t bytesize = 5;
    auto data = device0.Allocate(bytesize);

    // Transfer
    std::shared_ptr<void> trans_data = device0.TransferDataTo(device1, data, 0, bytesize);

    EXPECT_EQ(0, std::memcmp(data.get(), trans_data.get(), bytesize));
}

TEST_P(NativeBackendTransferTest, ArrayToDevice) {
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
        EXPECT_EQ(a.data().get(), b.data().get()) << "Array::ToDevice() must return alias when transferring to the same native device.";
    } else {
        EXPECT_NE(a.data().get(), b.data().get())
                << "Array::ToDevice() must not return alias when transferring to a different native device.";
    }
    ExpectArraysEqual(a, b);
}

}  // namespace
}  // namespace native
}  // namespace xchainer
