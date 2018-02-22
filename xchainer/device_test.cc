#include "xchainer/device.h"

#include <future>

#include <gtest/gtest.h>
#include "xchainer/backend.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_backend.h"
#include "xchainer/cuda/cuda_device.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/error.h"
#include "xchainer/native_backend.h"
#include "xchainer/native_device.h"

namespace xchainer {
namespace {

class DeviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        orig_ = internal::GetDefaultDeviceNoExcept();
        SetDefaultDevice(nullptr);
    }

    void TearDown() override { SetDefaultDevice(orig_); }

private:
    Device* orig_;
};

TEST_F(DeviceTest, Ctor) {
    NativeBackend native_backend;
    {
        NativeDevice device{native_backend, 0};
        EXPECT_EQ(&native_backend, &device.backend());
        EXPECT_EQ(0, device.index());
    }
    {
        NativeDevice device{native_backend, 1};
        EXPECT_EQ(&native_backend, &device.backend());
        EXPECT_EQ(1, device.index());
    }
}

TEST_F(DeviceTest, SetDefaultDevice) {
    ASSERT_THROW(GetDefaultDevice(), XchainerError);

    NativeBackend native_backend;
    NativeDevice native_device{native_backend, 0};
    SetDefaultDevice(&native_device);
    ASSERT_EQ(&native_device, &GetDefaultDevice());

#ifdef XCHAINER_ENABLE_CUDA
    cuda::CudaBackend cuda_backend;
    cuda::CudaDevice cuda_device{cuda_backend, 0};
    SetDefaultDevice(&cuda_device);
    ASSERT_EQ(&cuda_device, &GetDefaultDevice());
#endif  // XCHAINER_ENABLE_CUDA

    NativeBackend native_backend2;
    NativeDevice native_device2{native_backend2, 2};
    SetDefaultDevice(&native_device2);
    ASSERT_EQ(&native_device2, &GetDefaultDevice());
}

TEST_F(DeviceTest, ThreadLocal) {
    NativeBackend backend1;
    NativeDevice device1{backend1, 1};
    SetDefaultDevice(&device1);

    auto future = std::async(std::launch::async, [] {
        NativeBackend backend2;
        NativeDevice device2{backend2, 2};
        SetDefaultDevice(&device2);
        return &GetDefaultDevice();
    });
    ASSERT_NE(&GetDefaultDevice(), future.get());
}

TEST_F(DeviceTest, DeviceScopeCtor) {
    {
        // DeviceScope should work even if default device is not set
        NativeBackend backend;
        NativeDevice device{backend, 0};
        DeviceScope scope(device);
    }
    NativeBackend backend1;
    NativeDevice device1{backend1, 1};
    SetDefaultDevice(&device1);
    {
        NativeBackend backend2;
        NativeDevice device2{backend2, 2};
        DeviceScope scope(device2);
        EXPECT_EQ(&device2, &GetDefaultDevice());
    }
    ASSERT_EQ(&device1, &GetDefaultDevice());
    {
        DeviceScope scope;
        EXPECT_EQ(&device1, &GetDefaultDevice());
        NativeBackend backend2;
        NativeDevice device2{backend2, 2};
        SetDefaultDevice(&device2);
    }
    ASSERT_EQ(&device1, &GetDefaultDevice());
    NativeBackend backend2;
    NativeDevice device2{backend2, 2};
    {
        DeviceScope scope(device2);
        scope.Exit();
        EXPECT_EQ(&device1, &GetDefaultDevice());
        SetDefaultDevice(&device2);
        // not recovered here because the scope has already existed
    }
    ASSERT_EQ(&device2, &GetDefaultDevice());
}

}  // namespace
}  // namespace xchainer
