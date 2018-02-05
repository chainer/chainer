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
        orig_ = internal::GetCurrentDeviceNoExcept();
        SetCurrentDevice(kNullDevice);
    }

    void TearDown() override { SetCurrentDevice(orig_); }

private:
    Device orig_;
};

TEST_F(DeviceTest, MakeDevice) {
    auto native_backend = std::make_unique<NativeBackend>();
    Device expect = Device::MakeDevice("abcde", native_backend.get());
    Device actual = Device::MakeDevice("abcde", native_backend.get());
    EXPECT_EQ(expect, actual);

    EXPECT_THROW(Device::MakeDevice("12345678", native_backend.get()), DeviceError);
}

TEST_F(DeviceTest, SetCurrentDevice) {
    ASSERT_THROW(GetCurrentDevice(), XchainerError);

    auto native_backend = std::make_unique<NativeBackend>();
    auto native_device = Device::MakeDevice("cpu", native_backend.get());
    SetCurrentDevice(native_device);
    ASSERT_EQ(native_device, GetCurrentDevice());

#ifdef XCHAINER_ENABLE_CUDA
    auto cuda_backend = std::make_unique<cuda::CudaBackend>();
    auto cuda_device = Device::MakeDevice("cuda", cuda_backend.get());
    SetCurrentDevice(cuda_device);
    ASSERT_EQ(cuda_device, GetCurrentDevice());
#endif  // XCHAINER_ENABLE_CUDA

    auto native_backend2 = std::make_unique<NativeBackend>();
    auto native_device2 = Device::MakeDevice("cpu2", native_backend2.get());
    SetCurrentDevice(native_device2);
    ASSERT_EQ(native_device2, GetCurrentDevice());
}

TEST_F(DeviceTest, ThreadLocal) {
    auto backend1 = std::make_unique<NativeBackend>();
    auto device1 = Device::MakeDevice("cpu1", backend1.get());
    SetCurrentDevice(device1);

    auto future = std::async(std::launch::async, [] {
        auto backend2 = std::make_unique<NativeBackend>();
        auto device2 = Device::MakeDevice("cpu2", backend2.get());
        SetCurrentDevice(device2);
        return GetCurrentDevice();
    });
    ASSERT_NE(GetCurrentDevice(), future.get());
}

TEST_F(DeviceTest, DeviceScopeCtor) {
    {
        // DeviceScope should work even if current device is kNullDevice
        auto backend = std::make_unique<NativeBackend>();
        auto device = Device::MakeDevice("cpu", backend.get());
        DeviceScope scope(device);
    }
    auto backend1 = std::make_unique<NativeBackend>();
    auto device1 = Device::MakeDevice("cpu1", backend1.get());
    SetCurrentDevice(device1);
    {
        auto backend2 = std::make_unique<NativeBackend>();
        auto device2 = Device::MakeDevice("cpu2", backend2.get());
        DeviceScope scope(device2);
        EXPECT_EQ(device2, GetCurrentDevice());
    }
    ASSERT_EQ(device1, GetCurrentDevice());
    {
        DeviceScope scope;
        EXPECT_EQ(device1, GetCurrentDevice());
        auto backend2 = std::make_unique<NativeBackend>();
        auto device2 = Device::MakeDevice("cpu2", backend2.get());
        SetCurrentDevice(device2);
    }
    ASSERT_EQ(device1, GetCurrentDevice());
    auto backend2 = std::make_unique<NativeBackend>();
    auto device2 = Device::MakeDevice("cpu2", backend2.get());
    {
        DeviceScope scope(device2);
        scope.Exit();
        EXPECT_EQ(device1, GetCurrentDevice());
        SetCurrentDevice(device2);
        // not recovered here because the scope has already existed
    }
    ASSERT_EQ(device2, GetCurrentDevice());
}

}  // namespace
}  // namespace xchainer
