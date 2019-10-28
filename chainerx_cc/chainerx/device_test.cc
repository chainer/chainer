#include "chainerx/device.h"

#include <future>

#include <absl/types/optional.h>
#include <gtest/gtest.h>
#include <gsl/gsl>

#include "chainerx/backend.h"
#include "chainerx/context.h"
#ifdef CHAINERX_ENABLE_CUDA
#include "chainerx/cuda/cuda_backend.h"
#include "chainerx/cuda/cuda_device.h"
#endif  // CHAINERX_ENABLE_CUDA
#include "chainerx/error.h"
#include "chainerx/native/native_backend.h"
#include "chainerx/native/native_device.h"
#include "chainerx/testing/context_session.h"

namespace chainerx {
namespace {

class DeviceTest : public ::testing::Test {
protected:
    void SetUp() override { context_session_.emplace(); }

    void TearDown() override { context_session_.reset(); }

private:
    absl::optional<testing::ContextSession> context_session_;
};

TEST_F(DeviceTest, Ctor) {
    Context& ctx = GetDefaultContext();
    native::NativeBackend native_backend{ctx};
    {
        Device& device = native_backend.GetDevice(0);
        EXPECT_EQ(&native_backend, &device.backend());
        EXPECT_EQ(0, device.index());
    }
    {
        Device& device = native_backend.GetDevice(1);
        EXPECT_EQ(&native_backend, &device.backend());
        EXPECT_EQ(1, device.index());
    }
}

TEST_F(DeviceTest, SetDefaultDevice) {
    Context& ctx = GetDefaultContext();
    EXPECT_EQ(&ctx.GetDevice({"native", 0}), &GetDefaultDevice());

    native::NativeBackend native_backend{ctx};
    Device& native_device = native_backend.GetDevice(0);
    SetDefaultDevice(&native_device);
    ASSERT_EQ(&native_device, &GetDefaultDevice());

#ifdef CHAINERX_ENABLE_CUDA
    cuda::CudaBackend cuda_backend{ctx};
    Device& cuda_device = cuda_backend.GetDevice(0);
    SetDefaultDevice(&cuda_device);
    ASSERT_EQ(&cuda_device, &GetDefaultDevice());
#endif  // CHAINERX_ENABLE_CUDA

    Device& native_device2 = ctx.GetDevice({"native", 2});
    SetDefaultDevice(&native_device2);
    ASSERT_EQ(&native_device2, &GetDefaultDevice());
}

TEST_F(DeviceTest, ThreadLocal) {
    Context& ctx = GetDefaultContext();
    native::NativeBackend backend1{ctx};
    Device& device1 = backend1.GetDevice(1);
    SetDefaultDevice(&device1);

    native::NativeBackend backend2{ctx};
    Device& device2 = backend2.GetDevice(2);
    auto future = std::async(std::launch::async, [&ctx, &device2] {
        SetDefaultContext(&ctx);
        SetDefaultDevice(&device2);
        return &GetDefaultDevice();
    });
    ASSERT_NE(&GetDefaultDevice(), future.get());
}

TEST_F(DeviceTest, ThreadLocalDefault) {
    Context& ctx = GetDefaultContext();
    SetGlobalDefaultContext(&ctx);
    auto reset_global = gsl::finally([] { SetGlobalDefaultContext(nullptr); });

    Device& device = ctx.GetDevice({"native", 0});

    native::NativeBackend backend1{ctx};
    Device& device1 = backend1.GetDevice(1);
    SetDefaultDevice(&device1);
    auto future = std::async(std::launch::async, [] { return &GetDefaultDevice(); });

    EXPECT_EQ(&device, future.get());
}

TEST_F(DeviceTest, DeviceScopeCtor) {
    Context& ctx = GetDefaultContext();
    {
        // DeviceScope should work even if default device is not set
        native::NativeBackend backend{ctx};
        Device& device = backend.GetDevice(0);
        DeviceScope scope(device);
    }
    native::NativeBackend backend1{ctx};
    Device& device1 = backend1.GetDevice(1);
    SetDefaultDevice(&device1);
    {
        native::NativeBackend backend2{ctx};
        Device& device2 = backend2.GetDevice(2);
        DeviceScope scope(device2);
        EXPECT_EQ(&device2, &GetDefaultDevice());
    }
    ASSERT_EQ(&device1, &GetDefaultDevice());
    {
        DeviceScope scope;
        EXPECT_EQ(&device1, &GetDefaultDevice());
        native::NativeBackend backend2{ctx};
        Device& device2 = backend2.GetDevice(2);
        SetDefaultDevice(&device2);
    }
    ASSERT_EQ(&device1, &GetDefaultDevice());
    native::NativeBackend backend2{ctx};
    Device& device2 = backend2.GetDevice(2);
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
}  // namespace chainerx
