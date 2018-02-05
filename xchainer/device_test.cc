#include "xchainer/device.h"

#include <future>

#include <gtest/gtest.h>
#include "xchainer/error.h"

namespace xchainer {
namespace {

constexpr Device kCpu = {"cpu"};
constexpr Device kCuda = {"cuda"};

TEST(DeviceTest, MakeDevice) {
    Device expect = {"abcde"};
    Device actual = MakeDevice("abcde");
    EXPECT_EQ(expect, actual);

    EXPECT_THROW(MakeDevice("12345678"), DeviceError);
}

TEST(DeviceTest, SetCurrentDevice) {
    auto device = GetCurrentDevice();

    SetCurrentDevice("cpu");
    ASSERT_EQ(kCpu, GetCurrentDevice());

    SetCurrentDevice(kCuda);
    ASSERT_EQ(kCuda, GetCurrentDevice());

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

TEST(DeviceScopeTest, Ctor) {
    auto device = GetCurrentDevice();

    SetCurrentDevice("cuda");
    {
        DeviceScope scope("cpu");
        EXPECT_EQ(kCpu, GetCurrentDevice());
    }
    ASSERT_EQ(kCuda, GetCurrentDevice());
    {
        DeviceScope scope;
        EXPECT_EQ(kCuda, GetCurrentDevice());
        SetCurrentDevice("cpu");
    }
    ASSERT_EQ(kCuda, GetCurrentDevice());
    {
        DeviceScope scope("cpu");
        scope.Exit();
        EXPECT_EQ(kCuda, GetCurrentDevice());
        SetCurrentDevice("cpu");
        // not recovered here because the scope has already existed
    }
    ASSERT_EQ(kCpu, GetCurrentDevice());

    SetCurrentDevice(device);
}

}  // namespace
}  // namespace xchainer
