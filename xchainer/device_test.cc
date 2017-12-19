#include "xchainer/device.h"

#include <future>

#include <gtest/gtest.h>
#include "xchainer/error.h"

namespace xchainer {

TEST(DeviceTest, SetCurrentDevice) {
    Device cpu = {"cpu"};
    Device cuda = {"cuda"};
    auto device = GetCurrentDevice();

    SetCurrentDevice("cpu");
    ASSERT_EQ(cpu, GetCurrentDevice());

    SetCurrentDevice(cuda);
    ASSERT_EQ(cuda, GetCurrentDevice());

    ASSERT_THROW(SetCurrentDevice("invalid_device"), DeviceError);

    SetCurrentDevice(device);
}

TEST(DeviceTest, ThreadLocal) {
    auto device = GetCurrentDevice();

    SetCurrentDevice("cpu");
    auto future = std::async(std::launch::async, [] {
        SetCurrentDevice("cuda");
        return GetCurrentDevice();
    });
    ASSERT_NE(GetCurrentDevice(), future.get());

    SetCurrentDevice(device);
}

}  // namespace
