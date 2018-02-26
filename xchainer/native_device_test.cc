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

TEST(NativeDeviceTest, Allocate) {
    size_t bytesize = 3;
    NativeBackend backend;
    NativeDevice device{backend, 0};
    std::shared_ptr<void> ptr = device.Allocate(bytesize);

    EXPECT_NE(nullptr, ptr);
}

}  // namespace
}  // namespace xchainer
