#include "xchainer/device_list.h"

#include <gtest/gtest.h>
#include "xchainer/device.h"
#include "xchainer/native_backend.h"
#include "xchainer/native_device.h"

namespace xchainer {
namespace {

TEST(DeviceList, AddGet) {
    NativeBackend backend;
    NativeDevice device{&backend, 0};
    DeviceList device_list;

    auto device = std::make_unique<NativeDevice>(&backend, 0);
    device_list.AddDevice(device);
    EXPECT_TRUE(devlice_list.HasDevice(0));
    EXPECT_FALSE(devlice_list.HasDevice(1));
    EXPECT_EQ(*device.get(), devlice_list.GetDevice(0));
}

}  // namespace
}  // namespace xchainer
