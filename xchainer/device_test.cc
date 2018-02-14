#include "xchainer/device.h"

#include <future>

#include <gtest/gtest.h>
#include "xchainer/backend.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_backend.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/error.h"
#include "xchainer/native_backend.h"

namespace xchainer {
namespace {

class DeviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        orig_ = internal::GetDefaultDeviceNoExcept();
        SetDefaultDevice(internal::kNullDevice);
    }

    void TearDown() override { SetDefaultDevice(orig_); }

private:
    Device orig_;
};

TEST_F(DeviceTest, Ctor) {
    NativeBackend native_backend;
    Device expect{"abcde", &native_backend};
    Device actual{"abcde", &native_backend};
    EXPECT_EQ(expect, actual);

    EXPECT_THROW(Device("12345678", &native_backend), DeviceError);
}

TEST_F(DeviceTest, ToString) {
    EXPECT_EQ("Device(null)", internal::kNullDevice.ToString());

    NativeBackend native_backend;
    Device device{"cpu", &native_backend};
    EXPECT_EQ("Device('cpu')", device.ToString());
}

TEST_F(DeviceTest, SetDefaultDevice) {
    ASSERT_THROW(GetDefaultDevice(), XchainerError);

    NativeBackend native_backend;
    Device native_device{"cpu", &native_backend};
    SetDefaultDevice(native_device);
    ASSERT_EQ(native_device, GetDefaultDevice());

#ifdef XCHAINER_ENABLE_CUDA
    cuda::CudaBackend cuda_backend;
    Device cuda_device{"cuda", &cuda_backend};
    SetDefaultDevice(cuda_device);
    ASSERT_EQ(cuda_device, GetDefaultDevice());
#endif  // XCHAINER_ENABLE_CUDA

    NativeBackend native_backend2;
    Device native_device2{"cpu2", &native_backend2};
    SetDefaultDevice(native_device2);
    ASSERT_EQ(native_device2, GetDefaultDevice());
}

TEST_F(DeviceTest, ThreadLocal) {
    NativeBackend backend1;
    Device device1{"cpu1", &backend1};
    SetDefaultDevice(device1);

    auto future = std::async(std::launch::async, [] {
        NativeBackend backend2;
        Device device2{"cpu2", &backend2};
        SetDefaultDevice(device2);
        return GetDefaultDevice();
    });
    ASSERT_NE(GetDefaultDevice(), future.get());
}

TEST_F(DeviceTest, DeviceScopeCtor) {
    {
        // DeviceScope should work even if default device is kNullDevice
        NativeBackend backend;
        Device device{"cpu", &backend};
        DeviceScope scope(device);
    }
    NativeBackend backend1;
    Device device1{"cpu1", &backend1};
    SetDefaultDevice(device1);
    {
        NativeBackend backend2;
        Device device2{"cpu2", &backend2};
        DeviceScope scope(device2);
        EXPECT_EQ(device2, GetDefaultDevice());
    }
    ASSERT_EQ(device1, GetDefaultDevice());
    {
        DeviceScope scope;
        EXPECT_EQ(device1, GetDefaultDevice());
        NativeBackend backend2;
        Device device2{"cpu2", &backend2};
        SetDefaultDevice(device2);
    }
    ASSERT_EQ(device1, GetDefaultDevice());
    NativeBackend backend2;
    Device device2{"cpu2", &backend2};
    {
        DeviceScope scope(device2);
        scope.Exit();
        EXPECT_EQ(device1, GetDefaultDevice());
        SetDefaultDevice(device2);
        // not recovered here because the scope has already existed
    }
    ASSERT_EQ(device2, GetDefaultDevice());
}

}  // namespace
}  // namespace xchainer
