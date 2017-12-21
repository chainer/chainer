#include "xchainer/device.h"

#include <future>

#include <gtest/gtest.h>
#include "xchainer/error.h"

namespace xchainer {

// Device must be POD (plain old data) to be used as a thread local variable safely.
// ref. https://google.github.io/styleguide/cppguide.html#Static_and_Global_Variables
static_assert(std::is_pod<Device>::value, "Device must be POD");

TEST(DeviceTest, MakeDevice) {
    Device expect = {"abcde"};
    Device actual = MakeDevice("abcde");
    ASSERT_EQ(expect, actual);

    EXPECT_THROW(MakeDevice("12345678"), DeviceError);
}

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
