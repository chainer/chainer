#include "xchainer/array.h"

#include <initializer_list>

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA
#include <gtest/gtest.h>

#include "xchainer/array.h"
#include "xchainer/backend.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_backend.h"
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device.h"
#include "xchainer/memory.h"
#include "xchainer/native_backend.h"

namespace xchainer {
namespace {

// Check that Array data exists on the specified device
void ExpectDataExistsOnDevice(const Device& expected_device, const Array& array) {
    // Check device member of the Array
    EXPECT_EQ(expected_device, array.device());

    // Check device of data pointee
    if (expected_device.name() == "cpu") {
        EXPECT_FALSE(internal::IsPointerCudaMemory(array.data().get()));
    } else if (expected_device.name() == "cuda") {
        EXPECT_TRUE(internal::IsPointerCudaMemory(array.data().get()));
    } else {
        FAIL() << "invalid device";
    }
}

// Check that Arrays are created on the current device if no other devices are specified
void CheckDeviceFallback(const std::function<Array()>& create_array_func) {
    // Fallback to current device which is CPU
    {
        NativeBackend native_backend;
        Device cpu_device{"cpu", &native_backend};
        auto scope = std::make_unique<DeviceScope>(cpu_device);
        Array array = create_array_func();
        ExpectDataExistsOnDevice(cpu_device, array);
    }
#ifdef XCHAINER_ENABLE_CUDA
    // Fallback to current device which is GPU
    {
        cuda::CudaBackend cuda_backend;
        Device cuda_device{"cuda", &cuda_backend};
        auto scope = std::make_unique<DeviceScope>(cuda_device);
        Array array = create_array_func();
        ExpectDataExistsOnDevice(cuda_device, array);
    }
#endif
}

// Check that Arrays are created on the specified device, if specified, without taking into account the current device
void CheckDeviceExplicit(const std::function<Array(const Device& device)>& create_array_func) {
    NativeBackend native_backend;
    Device cpu_device{"cpu", &native_backend};

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
    cuda::CudaBackend cuda_backend;
    Device cuda_device{"cuda", &cuda_backend};

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

TEST(ArrayDeviceTest, FromBuffer) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;
    float raw_data[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
    std::shared_ptr<void> data(raw_data, [](float* ptr) {
        (void)ptr;  // unused
    });
    CheckDeviceFallback([&]() { return Array::FromBuffer(shape, dtype, data); });
    CheckDeviceExplicit([&](const Device& device) { return Array::FromBuffer(shape, dtype, data, device); });
}

TEST(ArrayDeviceTest, Empty) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;
    CheckDeviceFallback([&]() { return Array::Empty(shape, dtype); });
    CheckDeviceExplicit([&](const Device& device) { return Array::Empty(shape, dtype, device); });
}

TEST(ArrayDeviceTest, Full) {
    Shape shape({2, 3});
    Scalar scalar{2.f};
    Dtype dtype = Dtype::kFloat32;
    CheckDeviceFallback([&]() { return Array::Full(shape, scalar, dtype); });
    CheckDeviceFallback([&]() { return Array::Full(shape, scalar); });
    CheckDeviceExplicit([&](const Device& device) { return Array::Full(shape, scalar, dtype, device); });
    CheckDeviceExplicit([&](const Device& device) { return Array::Full(shape, scalar, device); });
}

TEST(ArrayDeviceTest, Zeros) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;
    CheckDeviceFallback([&]() { return Array::Zeros(shape, dtype); });
    CheckDeviceExplicit([&](const Device& device) { return Array::Zeros(shape, dtype, device); });
}

TEST(ArrayDeviceTest, Ones) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;
    CheckDeviceFallback([&]() { return Array::Ones(shape, dtype); });
    CheckDeviceExplicit([&](const Device& device) { return Array::Ones(shape, dtype, device); });
}

TEST(ArrayDeviceTest, EmptyLike) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;

    CheckDeviceFallback([&]() {
        Array array_orig = Array::Empty(shape, dtype);
        return Array::EmptyLike(array_orig);
    });
    CheckDeviceExplicit([&](const Device& device) {
        NativeBackend native_backend;
        Device cpu_device{"cpu", &native_backend};
        Array array_orig = Array::Empty(shape, dtype, cpu_device);
        return Array::EmptyLike(array_orig, device);
    });
}

TEST(ArrayDeviceTest, FullLike) {
    Shape shape({2, 3});
    Scalar scalar{2.f};
    Dtype dtype = Dtype::kFloat32;

    CheckDeviceFallback([&]() {
        Array array_orig = Array::Empty(shape, dtype);
        return Array::FullLike(array_orig, scalar);
    });
    CheckDeviceExplicit([&](const Device& device) {
        NativeBackend native_backend;
        Device cpu_device{"cpu", &native_backend};
        Array array_orig = Array::Empty(shape, dtype, cpu_device);
        return Array::FullLike(array_orig, scalar, device);
    });
}

TEST(ArrayDeviceTest, ZerosLike) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;

    CheckDeviceFallback([&]() {
        Array array_orig = Array::Empty(shape, dtype);
        return Array::ZerosLike(array_orig);
    });
    CheckDeviceExplicit([&](const Device& device) {
        NativeBackend native_backend;
        Device cpu_device{"cpu", &native_backend};
        Array array_orig = Array::Empty(shape, dtype, cpu_device);
        return Array::ZerosLike(array_orig, device);
    });
}

TEST(ArrayDeviceTest, OnesLike) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;

    CheckDeviceFallback([&]() {
        Array array_orig = Array::Empty(shape, dtype);
        return Array::OnesLike(array_orig);
    });
    CheckDeviceExplicit([&](const Device& device) {
        NativeBackend native_backend;
        Device cpu_device{"cpu", &native_backend};
        Array array_orig = Array::Empty(shape, dtype, cpu_device);
        return Array::OnesLike(array_orig, device);
    });
}

}  // namespace
}  // namespace xchainer
