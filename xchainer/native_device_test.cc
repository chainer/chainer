#include "xchainer/native_device.h"

#include <gtest/gtest.h>

#include "xchainer/context.h"
#include "xchainer/native_backend.h"

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

TEST(NativeDeviceTest, Ctor) {
    Context ctx;
    NativeBackend backend{ctx};
    {
        NativeDevice device{backend, 0};
        EXPECT_EQ(&backend, &device.backend());
        EXPECT_EQ(0, device.index());
    }
    {
        NativeDevice device{backend, 1};
        EXPECT_EQ(&backend, &device.backend());
        EXPECT_EQ(1, device.index());
    }
}

TEST(NativeDeviceTest, Allocate) {
    Context ctx;
    NativeBackend backend{ctx};
    NativeDevice device{backend, 0};

    size_t bytesize = 3;
    std::shared_ptr<void> ptr = device.Allocate(bytesize);
    EXPECT_NE(nullptr, ptr);
}

/*
TEST(NativeDeviceTest, MemoryCopy) {
    size_t size = 3;
    size_t bytesize = size * sizeof(float);
    float raw_data[] = {0, 1, 2};
    std::shared_ptr<void> src(raw_data, [](float* ptr) {
        (void)ptr;  // unused
    });

    Context ctx;
    NativeBackend backend{ctx};
    NativeDevice device{backend, 0};

    std::shared_ptr<void> dst = device.Allocate(bytesize);
    device.MemoryCopy(dst.get(), src.get(), bytesize);
    ExpectDataEqual<float>(src, dst, size);
}*/

TEST(NativeDeviceTest, FromBuffer) {
    size_t size = 3;
    size_t bytesize = size * sizeof(float);
    float raw_data[] = {0, 1, 2};
    std::shared_ptr<void> src(raw_data, [](float* ptr) {
        (void)ptr;  // unused
    });

    Context ctx;
    NativeBackend backend{ctx};
    NativeDevice device{backend, 0};

    std::shared_ptr<void> dst = device.FromBuffer(src, bytesize);
    EXPECT_EQ(src.get(), dst.get());
}

TEST(NativeDeviceTest, Synchronize) {
    Context ctx;
    NativeBackend backend{ctx};
    NativeDevice device{backend, 0};
    EXPECT_NO_THROW(device.Synchronize());
}

class NativeDeviceMemoryCopyTest : public ::testing::TestWithParam<::testing::tuple<std::string, std::string>> {};

INSTANTIATE_TEST_CASE_P(Devices, NativeDeviceMemoryCopyTest,
                        ::testing::Values(std::make_tuple("native:0", "native:0"),  // native:0 <-> native:0
                                          std::make_tuple("native:0", "native:1")   // native:0 <-> native:1
                                          ));

TEST_P(NativeDeviceMemoryCopyTest, MemoryCopyFrom) {
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

TEST_P(NativeDeviceMemoryCopyTest, MemoryCopyTo) {
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
}  // namespace xchainer
