#include "xchainer/device_id.h"

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

class DeviceIdTest : public ::testing::Test {
protected:
    void SetUp() override {
        orig_ = internal::GetDefaultDeviceIdNoExcept();
        SetDefaultDeviceId(internal::kNullDeviceId);
    }

    void TearDown() override { SetDefaultDeviceId(orig_); }

private:
    DeviceId orig_;
};

TEST_F(DeviceIdTest, Ctor) {
    NativeBackend native_backend;
    {
        DeviceId expect{&native_backend};
        DeviceId actual{&native_backend};
        EXPECT_EQ(expect, actual);
    }
    {
        DeviceId device_id{&native_backend};
        EXPECT_EQ(&native_backend, device_id.backend());
        EXPECT_EQ(0, device_id.index());
    }
    {
        DeviceId device_id{&native_backend, 1};
        EXPECT_EQ(&native_backend, device_id.backend());
        EXPECT_EQ(1, device_id.index());
    }
}

TEST_F(DeviceIdTest, ToString) {
    EXPECT_EQ("DeviceId(null)", internal::kNullDeviceId.ToString());

    NativeBackend native_backend;
    {
        DeviceId device_id{&native_backend};
        EXPECT_EQ("DeviceId('cpu', 0)", device_id.ToString());
    }
    {
        DeviceId device_id{&native_backend, 1};
        EXPECT_EQ("DeviceId('cpu', 1)", device_id.ToString());
    }
}

TEST_F(DeviceIdTest, SetDefaultDeviceId) {
    ASSERT_THROW(GetDefaultDeviceId(), XchainerError);

    NativeBackend native_backend;
    DeviceId native_device_id{&native_backend};
    SetDefaultDeviceId(native_device_id);
    ASSERT_EQ(native_device_id, GetDefaultDeviceId());

#ifdef XCHAINER_ENABLE_CUDA
    cuda::CudaBackend cuda_backend;
    DeviceId cuda_device_id{&cuda_backend};
    SetDefaultDeviceId(cuda_device_id);
    ASSERT_EQ(cuda_device_id, GetDefaultDeviceId());
#endif  // XCHAINER_ENABLE_CUDA

    NativeBackend native_backend2;
    DeviceId native_device_id2{&native_backend2, 2};
    SetDefaultDeviceId(native_device_id2);
    ASSERT_EQ(native_device_id2, GetDefaultDeviceId());
}

TEST_F(DeviceIdTest, ThreadLocal) {
    NativeBackend backend1;
    DeviceId device_id1{&backend1, 1};
    SetDefaultDeviceId(device_id1);

    auto future = std::async(std::launch::async, [] {
        NativeBackend backend2;
        DeviceId device_id2{&backend2, 2};
        SetDefaultDeviceId(device_id2);
        return GetDefaultDeviceId();
    });
    ASSERT_NE(GetDefaultDeviceId(), future.get());
}

TEST_F(DeviceIdTest, DeviceIdScopeCtor) {
    {
        // DeviceIdScope should work even if default device_id is kNullDeviceId
        NativeBackend backend;
        DeviceId device_id{&backend};
        DeviceIdScope scope(device_id);
    }
    NativeBackend backend1;
    DeviceId device_id1{&backend1, 1};
    SetDefaultDeviceId(device_id1);
    {
        NativeBackend backend2;
        DeviceId device_id2{&backend2, 2};
        DeviceIdScope scope(device_id2);
        EXPECT_EQ(device_id2, GetDefaultDeviceId());
    }
    ASSERT_EQ(device_id1, GetDefaultDeviceId());
    {
        DeviceIdScope scope;
        EXPECT_EQ(device_id1, GetDefaultDeviceId());
        NativeBackend backend2;
        DeviceId device_id2{&backend2, 2};
        SetDefaultDeviceId(device_id2);
    }
    ASSERT_EQ(device_id1, GetDefaultDeviceId());
    NativeBackend backend2;
    DeviceId device_id2{&backend2, 2};
    {
        DeviceIdScope scope(device_id2);
        scope.Exit();
        EXPECT_EQ(device_id1, GetDefaultDeviceId());
        SetDefaultDeviceId(device_id2);
        // not recovered here because the scope has already existed
    }
    ASSERT_EQ(device_id2, GetDefaultDeviceId());
}

}  // namespace
}  // namespace xchainer
