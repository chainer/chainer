#include "xchainer/array.h"

#include <cstddef>
#include <initializer_list>
#include <string>

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA
#include <gtest/gtest.h>

#include "xchainer/array.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device.h"
#include "xchainer/error.h"
#include "xchainer/memory.h"

namespace xchainer {
namespace {

template <typename T>
void ExpectDataEqual(const T* expected_data, const Array& actual) {
#ifdef XCHAINER_ENABLE_CUDA
    if (actual.device() == MakeDevice("cuda")) {
        cuda::CheckError(cudaDeviceSynchronize());
    }
#endif  // XCHAINER_ENABLE_CUDA
    auto total_size = actual.shape().total_size();
    const T* actual_data = static_cast<const T*>(actual.data().get());
    for (decltype(total_size) i = 0; i < total_size; i++) {
        EXPECT_EQ(expected_data[i], actual_data[i]) << "where i is " << i;
    }
}

template <typename T>
void ExpectDataEqual(T expected, const Array& actual) {
#ifdef XCHAINER_ENABLE_CUDA
    if (actual.device() == MakeDevice("cuda")) {
        cuda::CheckError(cudaDeviceSynchronize());
    }
#endif  // XCHAINER_ENABLE_CUDA
    auto total_size = actual.shape().total_size();
    const T* actual_data = static_cast<const T*>(actual.data().get());
    for (decltype(total_size) i = 0; i < total_size; i++) {
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual_data[i])) << "where i is " << i;
        } else {
            EXPECT_EQ(expected, actual_data[i]) << "where i is " << i;
        }
    }
}

void ExpectDataEqual(const Array& x, Dtype dtype, Scalar scalar) {
    VisitDtype(dtype, [&x, &scalar](auto pt) {
        using T = typename decltype(pt)::type;
        return ExpectDataEqual<T>(static_cast<T>(scalar), x);
    });
}

// Check that Array data exists on the specified device
void ExpectDataExistsOnDevice(const Device& expected_device, const Array& array) {
    // Check device of data pointee
    if (expected_device == MakeDevice("cpu")) {
        EXPECT_FALSE(internal::IsPointerCudaMemory(array.data().get()));
    } else if (expected_device == MakeDevice("cuda")) {
        EXPECT_TRUE(internal::IsPointerCudaMemory(array.data().get()));
    } else {
        FAIL() << "invalid device";
    }
}

// Common criteria for FromBuffer, Empty, Full, Ones, Zeros, etc.
void CheckCommonMembers(const Array& x, const Shape& shape, Dtype dtype, const Device& device) {
    EXPECT_NE(nullptr, x.data());
    EXPECT_EQ(shape, x.shape());
    EXPECT_EQ(shape.ndim(), x.ndim());
    EXPECT_EQ(shape.total_size(), x.total_size());
    EXPECT_EQ(dtype, x.dtype());
    EXPECT_EQ(device, x.device());
    EXPECT_TRUE(x.is_contiguous());
    EXPECT_EQ(0, x.offset());
    ExpectDataExistsOnDevice(device, x);
}

template <typename T>
void CheckFromBufferMembers(const Array& x, const Shape& shape, Dtype dtype, std::shared_ptr<T> data, const Device& device) {
    CheckCommonMembers(x, shape, dtype, device);
    EXPECT_EQ(int64_t{sizeof(T)}, x.element_bytes());
    EXPECT_EQ(shape.total_size() * int64_t{sizeof(T)}, x.total_bytes());
    ExpectDataEqual<T>(data.get(), x);
}

// Common criteria for EmptyLike, FullLike, OnesLike, ZerosLike, etc.
void CheckCommonLikeMembers(const Array& x, const Array& x_orig, const Device& device) {
    EXPECT_NE(nullptr, x.data());
    EXPECT_EQ(x_orig.shape(), x.shape());
    EXPECT_EQ(x_orig.dtype(), x.dtype());
    EXPECT_EQ(device, x.device());
    EXPECT_TRUE(x.is_contiguous());
    EXPECT_EQ(0, x.offset());
    ExpectDataExistsOnDevice(device, x);
}

class ArrayCreationTest : public ::testing::TestWithParam<::testing::tuple<std::string>> {
protected:
    void SetUp() override {
        std::string device_name = ::testing::get<0>(GetParam());
        device_ = MakeDevice(device_name);
    }

    Device& GetDevice() { return device_; }

    template <typename T>
    void CheckFromBuffer(const Shape& shape, Dtype dtype, std::initializer_list<T> raw_data) {
        auto device_scope = std::make_unique<DeviceScope>(device_);
        std::shared_ptr<T> data = std::make_unique<T[]>(shape.total_size());
        std::copy(raw_data.begin(), raw_data.end(), data.get());
        {
            Array x = Array::FromBuffer(shape, dtype, data);
            CheckFromBufferMembers(x, shape, dtype, data, device_);
        }
        {
            const Array x = Array::FromBuffer(shape, dtype, data);
            CheckFromBufferMembers(x, shape, dtype, data, device_);
        }
    }

    template <typename T>
    void CheckFromBuffer(const Shape& shape, Dtype dtype, std::initializer_list<T> raw_data, const Device& device) {
        std::shared_ptr<T> data = std::make_unique<T[]>(shape.total_size());
        std::copy(raw_data.begin(), raw_data.end(), data.get());
        {
            Array x = Array::FromBuffer(shape, dtype, data, device);
            CheckFromBufferMembers(x, shape, dtype, data, device);
        }
        {
            const Array x = Array::FromBuffer(shape, dtype, data, device);
            CheckFromBufferMembers(x, shape, dtype, data, device);
        }
    }

    void CheckEmpty(const Shape& shape, Dtype dtype) {
        auto device_scope = std::make_unique<DeviceScope>(device_);
        Array x = Array::Empty(shape, dtype);
        CheckCommonMembers(x, shape, dtype, device_);
    }

    void CheckEmpty(const Shape& shape, Dtype dtype, const Device& device) {
        Array x = Array::Empty(shape, dtype, device);
        CheckCommonMembers(x, shape, dtype, device);
    }

    void CheckFull(const Shape& shape, Scalar scalar, Dtype dtype) {
        auto device_scope = std::make_unique<DeviceScope>(device_);
        Array x = Array::Full(shape, scalar, dtype);
        CheckCommonMembers(x, shape, dtype, device_);
        ExpectDataEqual(x, dtype, scalar);
    }

    void CheckFull(const Shape& shape, Scalar scalar, Dtype dtype, const Device& device) {
        Array x = Array::Full(shape, scalar, dtype, device);
        CheckCommonMembers(x, shape, dtype, device);
        ExpectDataEqual(x, dtype, scalar);
    }

    void CheckFull(const Shape& shape, Scalar scalar) {
        auto device_scope = std::make_unique<DeviceScope>(device_);
        Array x = Array::Full(shape, scalar);
        CheckCommonMembers(x, shape, scalar.dtype(), device_);
        ExpectDataEqual(x, scalar.dtype(), scalar);
    }

    void CheckFull(const Shape& shape, Scalar scalar, const Device& device) {
        Array x = Array::Full(shape, scalar, device);
        CheckCommonMembers(x, shape, scalar.dtype(), device);
        ExpectDataEqual(x, scalar.dtype(), scalar);
    }

    void CheckZeros(const Shape& shape, Dtype dtype) {
        auto device_scope = std::make_unique<DeviceScope>(device_);
        Array x = Array::Zeros(shape, dtype);
        CheckCommonMembers(x, shape, dtype, device_);
        ExpectDataEqual(x, dtype, 0);
    }

    void CheckZeros(const Shape& shape, Dtype dtype, const Device& device) {
        Array x = Array::Zeros(shape, dtype, device);
        CheckCommonMembers(x, shape, dtype, device);
        ExpectDataEqual(x, dtype, 0);
    }

    void CheckOnes(const Shape& shape, Dtype dtype) {
        auto device_scope = std::make_unique<DeviceScope>(device_);
        Array x = Array::Ones(shape, dtype);
        CheckCommonMembers(x, shape, dtype, device_);
        ExpectDataEqual(x, dtype, 1);
    }

    void CheckOnes(const Shape& shape, Dtype dtype, const Device& device) {
        Array x = Array::Ones(shape, dtype, device);
        CheckCommonMembers(x, shape, dtype, device);
        ExpectDataEqual(x, dtype, 1);
    }

    void CheckEmptyLike(const Shape& shape, Dtype dtype) {
        auto device_scope = std::make_unique<DeviceScope>(device_);
        Array x_orig = Array::Empty(shape, dtype);
        Array x = Array::EmptyLike(x_orig);
        CheckCommonLikeMembers(x, x_orig, device_);
    }

    void CheckEmptyLike(const Shape& shape, Dtype dtype, const Device& device) {
        Array x_orig = Array::Empty(shape, dtype, device);
        Array x = Array::EmptyLike(x_orig, device);
        CheckCommonLikeMembers(x, x_orig, device);
    }

    void CheckFullLike(const Shape& shape, Scalar scalar) {
        auto device_scope = std::make_unique<DeviceScope>(device_);
        Array x_orig = Array::Empty(shape, scalar.dtype());
        Array x = Array::FullLike(x_orig, scalar);
        CheckCommonLikeMembers(x, x_orig, device_);
        ExpectDataEqual(x, scalar.dtype(), scalar);
    }

    void CheckFullLike(const Shape& shape, Scalar scalar, const Device& device) {
        Array x_orig = Array::Empty(shape, scalar.dtype(), device);
        Array x = Array::FullLike(x_orig, scalar, device);
        CheckCommonLikeMembers(x, x_orig, device);
        ExpectDataEqual(x, scalar.dtype(), scalar);
    }

    void CheckZerosLike(const Shape& shape, Dtype dtype) {
        auto device_scope = std::make_unique<DeviceScope>(device_);
        Array x_orig = Array::Empty(shape, dtype);
        Array x = Array::ZerosLike(x_orig);
        CheckCommonLikeMembers(x, x_orig, device_);
        ExpectDataEqual(x, dtype, 0);
    }

    void CheckZerosLike(const Shape& shape, Dtype dtype, const Device& device) {
        Array x_orig = Array::Empty(shape, dtype, device);
        Array x = Array::ZerosLike(x_orig, device);
        CheckCommonLikeMembers(x, x_orig, device);
        ExpectDataEqual(x, dtype, 0);
    }

    void CheckOnesLike(const Shape& shape, Dtype dtype) {
        auto device_scope = std::make_unique<DeviceScope>(device_);
        Array x_orig = Array::Empty(shape, dtype);
        Array x = Array::OnesLike(x_orig);
        CheckCommonLikeMembers(x, x_orig, device_);
        ExpectDataEqual(x, dtype, 1);
    }

    void CheckOnesLike(const Shape& shape, Dtype dtype, const Device& device) {
        Array x_orig = Array::Empty(shape, dtype, device);
        Array x = Array::OnesLike(x_orig, device);
        CheckCommonLikeMembers(x, x_orig, device);
        ExpectDataEqual(x, dtype, 1);
    }

private:
    Device device_;
};

TEST_P(ArrayCreationTest, FromBuffer) {
    Shape shape{3, 2};
    CheckFromBuffer<bool>(shape, Dtype::kBool, {true, false, false, true, false, true});
    CheckFromBuffer<bool>(shape, Dtype::kBool, {true, false, false, true, false, true}, GetDevice());
    CheckFromBuffer<int8_t>(shape, Dtype::kInt8, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<int8_t>(shape, Dtype::kInt8, {0, 1, 2, 3, 4, 5}, GetDevice());
    CheckFromBuffer<int16_t>(shape, Dtype::kInt16, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<int16_t>(shape, Dtype::kInt16, {0, 1, 2, 3, 4, 5}, GetDevice());
    CheckFromBuffer<int32_t>(shape, Dtype::kInt32, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<int32_t>(shape, Dtype::kInt32, {0, 1, 2, 3, 4, 5}, GetDevice());
    CheckFromBuffer<float>(shape, Dtype::kFloat32, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f});
    CheckFromBuffer<float>(shape, Dtype::kFloat32, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f}, GetDevice());
    CheckFromBuffer<double>(shape, Dtype::kFloat64, {0., 1., 2., 3., 4., 5.});
    CheckFromBuffer<double>(shape, Dtype::kFloat64, {0., 1., 2., 3., 4., 5.}, GetDevice());
}

#ifdef XCHAINER_ENABLE_CUDA
TEST_P(ArrayCreationTest, FromBufferFromNonManagedMemory) {
    Shape shape = {3, 2};
    Dtype dtype = Dtype::kBool;
    int64_t bytesize = shape.total_size() * sizeof(bool);

    void* raw_ptr = nullptr;
    cuda::CheckError(cudaMalloc(&raw_ptr, bytesize));
    auto data = std::shared_ptr<void>{raw_ptr, cudaFree};

    EXPECT_THROW(Array::FromBuffer(shape, dtype, data), XchainerError)
        << "FromBuffer must throw an exception if non-managed CUDA memory is given";
}
#endif  // XCHAINER_ENABLE_CUDA

TEST_P(ArrayCreationTest, Empty) {
    Shape shape({2, 3});
    for (const Dtype& dtype : GetAllDtypes()) {
        CheckEmpty(shape, dtype);
        CheckEmpty(shape, dtype, GetDevice());
    }
}

TEST_P(ArrayCreationTest, Full) {
    Shape shape({2, 3});
    CheckFull(shape, true);
    CheckFull(shape, true, GetDevice());
    CheckFull(shape, true, Dtype::kBool);
    CheckFull(shape, true, Dtype::kBool, GetDevice());
    CheckFull(shape, int8_t{2});
    CheckFull(shape, int8_t{2}, GetDevice());
    CheckFull(shape, int8_t{2}, Dtype::kInt8);
    CheckFull(shape, int8_t{2}, Dtype::kInt8, GetDevice());
    CheckFull(shape, int16_t{2});
    CheckFull(shape, int16_t{2}, GetDevice());
    CheckFull(shape, int16_t{2}, Dtype::kInt8);
    CheckFull(shape, int16_t{2}, Dtype::kInt8, GetDevice());
    CheckFull(shape, int32_t{2});
    CheckFull(shape, int32_t{2}, GetDevice());
    CheckFull(shape, int32_t{2}, Dtype::kInt8);
    CheckFull(shape, int32_t{2}, Dtype::kInt8, GetDevice());
    CheckFull(shape, uint8_t{2});
    CheckFull(shape, uint8_t{2}, GetDevice());
    CheckFull(shape, uint8_t{2}, Dtype::kInt8);
    CheckFull(shape, uint8_t{2}, Dtype::kInt8, GetDevice());
    CheckFull(shape, float{2.0f});
    CheckFull(shape, float{2.0f}, GetDevice());
    CheckFull(shape, float{2.0f}, Dtype::kInt8);
    CheckFull(shape, float{2.0f}, Dtype::kInt8, GetDevice());
    CheckFull(shape, double{2.0});
    CheckFull(shape, double{2.0}, GetDevice());
    CheckFull(shape, double{2.0}, Dtype::kInt8);
    CheckFull(shape, double{2.0}, Dtype::kInt8, GetDevice());
}

TEST_P(ArrayCreationTest, Zeros) {
    Shape shape({2, 3});
    for (const Dtype& dtype : GetAllDtypes()) {
        CheckZeros(shape, dtype);
        CheckZeros(shape, dtype, GetDevice());
    }
}

TEST_P(ArrayCreationTest, Ones) {
    Shape shape({2, 3});
    for (const Dtype& dtype : GetAllDtypes()) {
        CheckOnes(shape, dtype);
        CheckOnes(shape, dtype, GetDevice());
    }
}

TEST_P(ArrayCreationTest, EmptyLike) {
    Shape shape({2, 3});
    for (const Dtype& dtype : GetAllDtypes()) {
        CheckEmptyLike(shape, dtype);
        CheckEmptyLike(shape, dtype, GetDevice());
    }
}

TEST_P(ArrayCreationTest, FullLike) {
    Shape shape({2, 3});
    CheckFullLike(shape, true);
    CheckFullLike(shape, true, GetDevice());
    CheckFullLike(shape, int8_t{2});
    CheckFullLike(shape, int8_t{2}, GetDevice());
    CheckFullLike(shape, int16_t{2});
    CheckFullLike(shape, int16_t{2}, GetDevice());
    CheckFullLike(shape, int32_t{2});
    CheckFullLike(shape, int32_t{2}, GetDevice());
    CheckFullLike(shape, uint8_t{2});
    CheckFullLike(shape, uint8_t{2}, GetDevice());
    CheckFullLike(shape, float{2.0f});
    CheckFullLike(shape, float{2.0f}, GetDevice());
    CheckFullLike(shape, double{2.0});
    CheckFullLike(shape, double{2.0}, GetDevice());
}

TEST_P(ArrayCreationTest, ZerosLike) {
    Shape shape({2, 3});
    for (const Dtype& dtype : GetAllDtypes()) {
        CheckZerosLike(shape, dtype);
        CheckZerosLike(shape, dtype, GetDevice());
    }
}

TEST_P(ArrayCreationTest, OnesLike) {
    Shape shape({2, 3});
    for (const Dtype& dtype : GetAllDtypes()) {
        CheckOnesLike(shape, dtype);
        CheckOnesLike(shape, dtype, GetDevice());
    }
}

INSTANTIATE_TEST_CASE_P(ForEachDevice, ArrayCreationTest, ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                                                              std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                                                              std::string{"cpu"}));

}  // namespace
}  // namespace xchainer
