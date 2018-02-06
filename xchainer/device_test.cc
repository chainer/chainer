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
        SetCurrentDevice(internal::kNullDevice);
    }

    void TearDown() override { SetCurrentDevice(orig_); }

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
    EXPECT_EQ(internal::kNullDevice.ToString(), "<Device()>");

    NativeBackend native_backend;
    Device device{"cpu", &native_backend};
    std::ostringstream os;
    os << "<Device('cpu', " << &native_backend << ")>";
    EXPECT_EQ(device.ToString(), os.str());
}

TEST_F(DeviceTest, SetCurrentDevice) {
    ASSERT_THROW(GetCurrentDevice(), XchainerError);

    NativeBackend native_backend;
    Device native_device{"cpu", &native_backend};
    SetCurrentDevice(native_device);
    ASSERT_EQ(native_device, GetCurrentDevice());

#ifdef XCHAINER_ENABLE_CUDA
    cuda::CudaBackend cuda_backend;
    Device cuda_device{"cuda", &cuda_backend};
    SetCurrentDevice(cuda_device);
    ASSERT_EQ(cuda_device, GetCurrentDevice());
#endif  // XCHAINER_ENABLE_CUDA

    NativeBackend native_backend2;
    Device native_device2{"cpu2", &native_backend2};
    SetCurrentDevice(native_device2);
    ASSERT_EQ(native_device2, GetCurrentDevice());
}

TEST_F(DeviceTest, ThreadLocal) {
    NativeBackend backend1;
    Device device1{"cpu1", &backend1};
    SetCurrentDevice(device1);

    auto future = std::async(std::launch::async, [] {
        NativeBackend backend2;
        Device device2{"cpu2", &backend2};
        SetCurrentDevice(device2);
        return GetCurrentDevice();
    });
    ASSERT_NE(GetCurrentDevice(), future.get());
}

TEST_F(DeviceTest, DeviceScopeCtor) {
    {
        // DeviceScope should work even if current device is kNullDevice
        NativeBackend backend;
        Device device{"cpu", &backend};
        DeviceScope scope(device);
    }
    NativeBackend backend1;
    Device device1{"cpu1", &backend1};
    SetCurrentDevice(device1);
    {
        NativeBackend backend2;
        Device device2{"cpu2", &backend2};
        DeviceScope scope(device2);
        EXPECT_EQ(device2, GetCurrentDevice());
    }
    ASSERT_EQ(device1, GetCurrentDevice());
    {
        DeviceScope scope;
        EXPECT_EQ(device1, GetCurrentDevice());
        NativeBackend backend2;
        Device device2{"cpu2", &backend2};
        SetCurrentDevice(device2);
    }
    ASSERT_EQ(device1, GetCurrentDevice());
    NativeBackend backend2;
    Device device2{"cpu2", &backend2};
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
