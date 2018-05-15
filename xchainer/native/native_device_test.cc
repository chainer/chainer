#include "xchainer/native/native_device.h"

#include <gtest/gtest.h>

#include "xchainer/context.h"
#include "xchainer/native/native_backend.h"

#include "xchainer/testing/array.h"

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

TEST(NativeDeviceTest, MakeDataFromForeignPointer) {
    Context ctx;
    NativeBackend backend{ctx};
    NativeDevice device{backend, 0};

    size_t bytesize = 3;
    std::shared_ptr<void> data = device.Allocate(bytesize);
    EXPECT_EQ(data.get(), device.MakeDataFromForeignPointer(data).get());
}

TEST(NativeDeviceTest, FromHostMemory) {
    size_t size = 3;
    size_t bytesize = size * sizeof(float);
    float raw_data[] = {0, 1, 2};
    std::shared_ptr<void> src(raw_data, [](const float* ptr) {
        (void)ptr;  // unused
    });

    Context ctx;
    NativeBackend backend{ctx};
    NativeDevice device{backend, 0};

    std::shared_ptr<void> dst = device.FromHostMemory(src, bytesize);
    EXPECT_EQ(src.get(), dst.get());
}

TEST(NativeDeviceTest, Synchronize) {
    Context ctx;
    NativeBackend backend{ctx};
    NativeDevice device{backend, 0};
    EXPECT_NO_THROW(device.Synchronize());
}

// TODO(hvy): Remove me
TEST(NativeDeviceTest, Im2Col) {
    Context ctx;
    NativeBackend backend{ctx};
    NativeDevice device{backend, 0};
    ContextScope context_scope(ctx);

    //Array x = testing::BuildArray({1, 1, 4, 1}).WithLinearData<float>();
    //Array y = Im2Col(x, {2, 1}, {1, 1}, {0, 0}, false);

    Array x = testing::BuildArray({1, 1, 7, 6}).WithLinearData<float>();
    Array y = Im2Col(x, {3, 3}, {2, 2}, {0, 0}, false);

    //Array x = testing::BuildArray({1, 1, 4, 3, 5}).WithLinearData<float>();
    //Array y = Im2Col(x, {3, 3, 3}, {2, 2, 2}, {0, 0, 0}, false);

    std::cout << y << std::endl;
}

}  // namespace
}  // namespace native
}  // namespace xchainer
