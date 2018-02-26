#include "xchainer/array.h"

#include <initializer_list>

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA
#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/backend.h"
#include "xchainer/context.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_backend.h"
#include "xchainer/cuda/cuda_device.h"
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device.h"
#include "xchainer/memory.h"
#include "xchainer/native_backend.h"
#include "xchainer/native_device.h"
#include "xchainer/testing/context_session.h"

namespace xchainer {
namespace {

class ArrayDeviceTest : public ::testing::Test {
protected:
    void SetUp() override { context_session_.emplace(); }

    void TearDown() override { context_session_.reset(); }

private:
    nonstd::optional<testing::ContextSession> context_session_;
};

// Check that Array data exists on the specified device
void ExpectDataExistsOnDevice(const Device& expected_device, const Array& array) {
    // Check device member of the Array
    EXPECT_EQ(&expected_device, &array.device());

    // Check device of data pointee
    if (expected_device.backend().GetName() == "native") {
        EXPECT_FALSE(internal::IsPointerCudaMemory(array.data().get()));
    } else if (expected_device.backend().GetName() == "cuda") {
        EXPECT_TRUE(internal::IsPointerCudaMemory(array.data().get()));
    } else {
        FAIL() << "invalid device";
    }
}

// Check that Arrays are created on the default device if no other devices are specified
void CheckDeviceFallback(const std::function<Array()>& create_array_func) {
    // Fallback to default device which is CPU
    {
        Context& ctx = GetDefaultContext();
        NativeBackend native_backend{ctx};
        NativeDevice cpu_device{native_backend, 0};
        auto scope = std::make_unique<DeviceScope>(cpu_device);
        Array array = create_array_func();
        ExpectDataExistsOnDevice(cpu_device, array);
    }
#ifdef XCHAINER_ENABLE_CUDA
    // Fallback to default device which is GPU
    {
        Context& ctx = GetDefaultContext();
        cuda::CudaBackend cuda_backend{ctx};
        cuda::CudaDevice cuda_device{cuda_backend, 0};
        auto scope = std::make_unique<DeviceScope>(cuda_device);
        Array array = create_array_func();
        ExpectDataExistsOnDevice(cuda_device, array);
    }
#endif
}

// Check that Arrays are created on the specified device, if specified, without taking into account the default device
void CheckDeviceExplicit(const std::function<Array(Device& device)>& create_array_func) {
    Context& ctx = GetDefaultContext();
    NativeBackend native_backend{ctx};
    NativeDevice cpu_device{native_backend, 0};

    // Explicitly create on CPU
    {
        Array array = create_array_func(cpu_device);
        ExpectDataExistsOnDevice(cpu_device, array);
    }
    {
        auto scope = std::make_unique<DeviceScope>(cpu_device);
        Array array = create_array_func(cpu_device);
        ExpectDataExistsOnDevice(cpu_device, array);
    }
#ifdef XCHAINER_ENABLE_CUDA
    cuda::CudaBackend cuda_backend{ctx};
    cuda::CudaDevice cuda_device{cuda_backend, 0};

    {
        auto scope = std::make_unique<DeviceScope>(cuda_device);
        Array array = create_array_func(cpu_device);
        ExpectDataExistsOnDevice(cpu_device, array);
    }
    // Explicitly create on GPU
    {
        Array array = create_array_func(cuda_device);
        ExpectDataExistsOnDevice(cuda_device, array);
    }
    {
        auto scope = std::make_unique<DeviceScope>(cpu_device);
        Array array = create_array_func(cuda_device);
        ExpectDataExistsOnDevice(cuda_device, array);
    }
    {
        auto scope = std::make_unique<DeviceScope>(cuda_device);
        Array array = create_array_func(cuda_device);
        ExpectDataExistsOnDevice(cuda_device, array);
    }
#endif  // XCHAINER_ENABLE_CUDA
}

TEST_F(ArrayDeviceTest, FromBuffer) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;
    float raw_data[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
    std::shared_ptr<void> data(raw_data, [](float* ptr) {
        (void)ptr;  // unused
    });
    CheckDeviceFallback([&]() { return Array::FromBuffer(shape, dtype, data); });
    CheckDeviceExplicit([&](Device& device) { return Array::FromBuffer(shape, dtype, data, device); });
}

TEST_F(ArrayDeviceTest, Empty) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;
    CheckDeviceFallback([&]() { return Array::Empty(shape, dtype); });
    CheckDeviceExplicit([&](Device& device) { return Array::Empty(shape, dtype, device); });
}

TEST_F(ArrayDeviceTest, Full) {
    Shape shape({2, 3});
    Scalar scalar{2.f};
    Dtype dtype = Dtype::kFloat32;
    CheckDeviceFallback([&]() { return Array::Full(shape, scalar, dtype); });
    CheckDeviceFallback([&]() { return Array::Full(shape, scalar); });
    CheckDeviceExplicit([&](Device& device) { return Array::Full(shape, scalar, dtype, device); });
    CheckDeviceExplicit([&](Device& device) { return Array::Full(shape, scalar, device); });
}

TEST_F(ArrayDeviceTest, Zeros) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;
    CheckDeviceFallback([&]() { return Array::Zeros(shape, dtype); });
    CheckDeviceExplicit([&](Device& device) { return Array::Zeros(shape, dtype, device); });
}

TEST_F(ArrayDeviceTest, Ones) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;
    CheckDeviceFallback([&]() { return Array::Ones(shape, dtype); });
    CheckDeviceExplicit([&](Device& device) { return Array::Ones(shape, dtype, device); });
}

TEST_F(ArrayDeviceTest, EmptyLike) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;

    CheckDeviceFallback([&]() {
        Array array_orig = Array::Empty(shape, dtype);
        return Array::EmptyLike(array_orig);
    });
    CheckDeviceExplicit([&](Device& device) {
        NativeBackend native_backend{device.context()};
        NativeDevice cpu_device{native_backend, 0};
        Array array_orig = Array::Empty(shape, dtype, cpu_device);
        return Array::EmptyLike(array_orig, device);
    });
}

TEST_F(ArrayDeviceTest, FullLike) {
    Shape shape({2, 3});
    Scalar scalar{2.f};
    Dtype dtype = Dtype::kFloat32;

    CheckDeviceFallback([&]() {
        Array array_orig = Array::Empty(shape, dtype);
        return Array::FullLike(array_orig, scalar);
    });
    CheckDeviceExplicit([&](Device& device) {
        NativeBackend native_backend{device.context()};
        NativeDevice cpu_device{native_backend, 0};
        Array array_orig = Array::Empty(shape, dtype, cpu_device);
        return Array::FullLike(array_orig, scalar, device);
    });
}

TEST_F(ArrayDeviceTest, ZerosLike) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;

    CheckDeviceFallback([&]() {
        Array array_orig = Array::Empty(shape, dtype);
        return Array::ZerosLike(array_orig);
    });
    CheckDeviceExplicit([&](Device& device) {
        NativeBackend native_backend{device.context()};
        NativeDevice cpu_device{native_backend, 0};
        Array array_orig = Array::Empty(shape, dtype, cpu_device);
        return Array::ZerosLike(array_orig, device);
    });
}

TEST_F(ArrayDeviceTest, OnesLike) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;

    CheckDeviceFallback([&]() {
        Array array_orig = Array::Empty(shape, dtype);
        return Array::OnesLike(array_orig);
    });
    CheckDeviceExplicit([&](Device& device) {
        NativeBackend native_backend{device.context()};
        NativeDevice cpu_device{native_backend, 0};
        Array array_orig = Array::Empty(shape, dtype, cpu_device);
        return Array::OnesLike(array_orig, device);
    });
}

}  // namespace
}  // namespace xchainer
