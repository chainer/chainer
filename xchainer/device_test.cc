#include "xchainer/device.h"

#include <future>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/backend.h"
#include "xchainer/context.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_backend.h"
#include "xchainer/cuda/cuda_device.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/error.h"
#include "xchainer/native_backend.h"
#include "xchainer/native_device.h"
#include "xchainer/testing/context_session.h"

namespace xchainer {
namespace {

class DeviceTest : public ::testing::Test {
protected:
    void SetUp() override { context_session_.emplace(); }

    void TearDown() override { context_session_.reset(); }

private:
    nonstd::optional<testing::ContextSession> context_session_;
};

TEST_F(DeviceTest, Ctor) {
    Context& ctx = GetDefaultContext();
    NativeBackend native_backend{ctx};
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

    Context& ctx = GetDefaultContext();
    NativeBackend native_backend{ctx};
    NativeDevice native_device{native_backend, 0};
    SetDefaultDevice(&native_device);
    ASSERT_EQ(&native_device, &GetDefaultDevice());

#ifdef XCHAINER_ENABLE_CUDA
    cuda::CudaBackend cuda_backend{ctx};
    cuda::CudaDevice cuda_device{cuda_backend, 0};
    SetDefaultDevice(&cuda_device);
    ASSERT_EQ(&cuda_device, &GetDefaultDevice());
#endif  // XCHAINER_ENABLE_CUDA

    NativeBackend native_backend2{ctx};
    NativeDevice native_device2{native_backend2, 2};
    SetDefaultDevice(&native_device2);
    ASSERT_EQ(&native_device2, &GetDefaultDevice());
}

TEST_F(DeviceTest, ThreadLocal) {
    Context& ctx = GetDefaultContext();
    NativeBackend backend1{ctx};
    NativeDevice device1{backend1, 1};
    SetDefaultDevice(&device1);

    NativeBackend backend2{ctx};
    NativeDevice device2{backend2, 2};
    auto future = std::async(std::launch::async, [&ctx, &device2] {
        SetDefaultContext(&ctx);
        SetDefaultDevice(&device2);
        return &GetDefaultDevice();
    });
    ASSERT_NE(&GetDefaultDevice(), future.get());
}

TEST_F(DeviceTest, DeviceScopeCtor) {
    Context& ctx = GetDefaultContext();
    {
        // DeviceScope should work even if default device is not set
        NativeBackend backend{ctx};
        NativeDevice device{backend, 0};
        DeviceScope scope(device);
    }
    NativeBackend backend1{ctx};
    NativeDevice device1{backend1, 1};
    SetDefaultDevice(&device1);
    {
        NativeBackend backend2{ctx};
        NativeDevice device2{backend2, 2};
        DeviceScope scope(device2);
        EXPECT_EQ(&device2, &GetDefaultDevice());
    }
    ASSERT_EQ(&device1, &GetDefaultDevice());
    {
        DeviceScope scope;
        EXPECT_EQ(&device1, &GetDefaultDevice());
        NativeBackend backend2{ctx};
        NativeDevice device2{backend2, 2};
        SetDefaultDevice(&device2);
    }
    ASSERT_EQ(&device1, &GetDefaultDevice());
    NativeBackend backend2{ctx};
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
