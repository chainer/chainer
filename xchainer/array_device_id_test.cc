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
#include "xchainer/device_id.h"
#include "xchainer/memory.h"
#include "xchainer/native_backend.h"

namespace xchainer {
namespace {

class ArrayDeviceIdTest : public ::testing::Test {
protected:
    void SetUp() override {
        orig_ = internal::GetDefaultDeviceIdNoExcept();
        SetDefaultDeviceId(internal::kNullDeviceId);
    }

    void TearDown() override { SetDefaultDeviceId(orig_); }

private:
    DeviceId orig_;
};

// Check that Array data exists on the specified device_id
void ExpectDataExistsOnDeviceId(const DeviceId& expected_device_id, const Array& array) {
    // Check device_id member of the Array
    EXPECT_EQ(expected_device_id, array.device_id());

    // Check device_id of data pointee
    if (expected_device_id.backend()->GetName() == "native") {
        EXPECT_FALSE(internal::IsPointerCudaMemory(array.data().get()));
    } else if (expected_device_id.backend()->GetName() == "cuda") {
        EXPECT_TRUE(internal::IsPointerCudaMemory(array.data().get()));
    } else {
        FAIL() << "invalid device_id";
    }
}

// Check that Arrays are created on the default device_id if no other device_ids are specified
void CheckDeviceIdFallback(const std::function<Array()>& create_array_func) {
    // Fallback to default device_id which is CPU
    {
        NativeBackend native_backend;
        DeviceId cpu_device_id{&native_backend};
        auto scope = std::make_unique<DeviceScope>(cpu_device_id);
        Array array = create_array_func();
        ExpectDataExistsOnDeviceId(cpu_device_id, array);
    }
#ifdef XCHAINER_ENABLE_CUDA
    // Fallback to default device_id which is GPU
    {
        cuda::CudaBackend cuda_backend;
        DeviceId cuda_device_id{&cuda_backend};
        auto scope = std::make_unique<DeviceScope>(cuda_device_id);
        Array array = create_array_func();
        ExpectDataExistsOnDeviceId(cuda_device_id, array);
    }
#endif
}

// Check that Arrays are created on the specified device_id, if specified, without taking into account the default device_id
void CheckDeviceIdExplicit(const std::function<Array(const DeviceId& device_id)>& create_array_func) {
    NativeBackend native_backend;
    DeviceId cpu_device_id{&native_backend};

    // Explicitly create on CPU
    {
        Array array = create_array_func(cpu_device_id);
        ExpectDataExistsOnDeviceId(cpu_device_id, array);
    }
    {
        auto scope = std::make_unique<DeviceScope>(cpu_device_id);
        Array array = create_array_func(cpu_device_id);
        ExpectDataExistsOnDeviceId(cpu_device_id, array);
    }
#ifdef XCHAINER_ENABLE_CUDA
    cuda::CudaBackend cuda_backend;
    DeviceId cuda_device_id{&cuda_backend};

    {
        auto scope = std::make_unique<DeviceScope>(cuda_device_id);
        Array array = create_array_func(cpu_device_id);
        ExpectDataExistsOnDeviceId(cpu_device_id, array);
    }
    // Explicitly create on GPU
    {
        Array array = create_array_func(cuda_device_id);
        ExpectDataExistsOnDeviceId(cuda_device_id, array);
    }
    {
        auto scope = std::make_unique<DeviceScope>(cpu_device_id);
        Array array = create_array_func(cuda_device_id);
        ExpectDataExistsOnDeviceId(cuda_device_id, array);
    }
    {
        auto scope = std::make_unique<DeviceScope>(cuda_device_id);
        Array array = create_array_func(cuda_device_id);
        ExpectDataExistsOnDeviceId(cuda_device_id, array);
    }
#endif  // XCHAINER_ENABLE_CUDA
}

TEST_F(ArrayDeviceIdTest, FromBuffer) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;
    float raw_data[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
    std::shared_ptr<void> data(raw_data, [](float* ptr) {
        (void)ptr;  // unused
    });
    CheckDeviceIdFallback([&]() { return Array::FromBuffer(shape, dtype, data); });
    CheckDeviceIdExplicit([&](const DeviceId& device_id) { return Array::FromBuffer(shape, dtype, data, device_id); });
}

TEST_F(ArrayDeviceIdTest, Empty) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;
    CheckDeviceIdFallback([&]() { return Array::Empty(shape, dtype); });
    CheckDeviceIdExplicit([&](const DeviceId& device_id) { return Array::Empty(shape, dtype, device_id); });
}

TEST_F(ArrayDeviceIdTest, Full) {
    Shape shape({2, 3});
    Scalar scalar{2.f};
    Dtype dtype = Dtype::kFloat32;
    CheckDeviceIdFallback([&]() { return Array::Full(shape, scalar, dtype); });
    CheckDeviceIdFallback([&]() { return Array::Full(shape, scalar); });
    CheckDeviceIdExplicit([&](const DeviceId& device_id) { return Array::Full(shape, scalar, dtype, device_id); });
    CheckDeviceIdExplicit([&](const DeviceId& device_id) { return Array::Full(shape, scalar, device_id); });
}

TEST_F(ArrayDeviceIdTest, Zeros) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;
    CheckDeviceIdFallback([&]() { return Array::Zeros(shape, dtype); });
    CheckDeviceIdExplicit([&](const DeviceId& device_id) { return Array::Zeros(shape, dtype, device_id); });
}

TEST_F(ArrayDeviceIdTest, Ones) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;
    CheckDeviceIdFallback([&]() { return Array::Ones(shape, dtype); });
    CheckDeviceIdExplicit([&](const DeviceId& device_id) { return Array::Ones(shape, dtype, device_id); });
}

TEST_F(ArrayDeviceIdTest, EmptyLike) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;

    CheckDeviceIdFallback([&]() {
        Array array_orig = Array::Empty(shape, dtype);
        return Array::EmptyLike(array_orig);
    });
    CheckDeviceIdExplicit([&](const DeviceId& device_id) {
        NativeBackend native_backend;
        DeviceId cpu_device_id{&native_backend};
        Array array_orig = Array::Empty(shape, dtype, cpu_device_id);
        return Array::EmptyLike(array_orig, device_id);
    });
}

TEST_F(ArrayDeviceIdTest, FullLike) {
    Shape shape({2, 3});
    Scalar scalar{2.f};
    Dtype dtype = Dtype::kFloat32;

    CheckDeviceIdFallback([&]() {
        Array array_orig = Array::Empty(shape, dtype);
        return Array::FullLike(array_orig, scalar);
    });
    CheckDeviceIdExplicit([&](const DeviceId& device_id) {
        NativeBackend native_backend;
        DeviceId cpu_device_id{&native_backend};
        Array array_orig = Array::Empty(shape, dtype, cpu_device_id);
        return Array::FullLike(array_orig, scalar, device_id);
    });
}

TEST_F(ArrayDeviceIdTest, ZerosLike) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;

    CheckDeviceIdFallback([&]() {
        Array array_orig = Array::Empty(shape, dtype);
        return Array::ZerosLike(array_orig);
    });
    CheckDeviceIdExplicit([&](const DeviceId& device_id) {
        NativeBackend native_backend;
        DeviceId cpu_device_id{&native_backend};
        Array array_orig = Array::Empty(shape, dtype, cpu_device_id);
        return Array::ZerosLike(array_orig, device_id);
    });
}

TEST_F(ArrayDeviceIdTest, OnesLike) {
    Shape shape({2, 3});
    Dtype dtype = Dtype::kFloat32;

    CheckDeviceIdFallback([&]() {
        Array array_orig = Array::Empty(shape, dtype);
        return Array::OnesLike(array_orig);
    });
    CheckDeviceIdExplicit([&](const DeviceId& device_id) {
        NativeBackend native_backend;
        DeviceId cpu_device_id{&native_backend};
        Array array_orig = Array::Empty(shape, dtype, cpu_device_id);
        return Array::OnesLike(array_orig, device_id);
    });
}

}  // namespace
}  // namespace xchainer
