#include "xchainer/native_device.h"

#include <gtest/gtest.h>

#include "xchainer/native_backend.h"

namespace xchainer {
namespace {

TEST(NativeDeviceTest, Ctor) {
    NativeBackend backend;

    {
        NativeDevice device{backend, 0};
        EXPECT_EQ(&backend, &deivce.backend());
        EXPECT_EQ(0, device.index());
    }
    {
        NativeDevice device{backend, 1};
        EXPECT_EQ(&backend, &deivce.backend());
        EXPECT_EQ(1, device.index());
    }
}

TEST(NativeDeviceTest, FromBuffer) {
    size_t size = 3;
    size_t bytesize = size * sizeof(float);
    float raw_data[] = {0, 1, 2};
    std::shared_ptr<void> src(raw_data, [](float* ptr) {
        (void)ptr;  // unused
    });

    NativeDevice backend;
    NativeDevice device{backend, 0};

    std::shared_ptr<void> dst = device.FromBuffer(src, bytesize);
    EXPECT_EQ(src.get(), dst.get());
}

}  // namespace
}  // namespace xchainer
