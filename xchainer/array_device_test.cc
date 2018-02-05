#include "xchainer/array.h"

#include <cassert>
#include <initializer_list>
#include <string>

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA
#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device.h"
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

class ArrayDeviceTest : public ::testing::TestWithParam<::testing::tuple<nonstd::optional<std::string>, std::string>> {
protected:
    void SetUp() override {
        // Device scope is optional since we want to test device specified Array creation where no current device is set
        nonstd::optional<std::string> device_scope_name = ::testing::get<0>(GetParam());
        std::string device_name = ::testing::get<1>(GetParam());

        // Enter device scope
        if (device_scope_name) {
            device_scope_ = std::make_unique<DeviceScope>(device_scope_name.value());
        }

        // Create the device in which Arrays will be explicitly created
        device_ = MakeDevice(device_name);
    }

    void TearDown() override {
        // Exit device scope if it was entered
        if (device_scope_) {
            device_scope_.reset();
        }
    }

    template <typename T>
    void CheckFromBuffer(const Shape& shape, Dtype dtype, std::initializer_list<T> raw_data) {
        assert(shape.total_size() == static_cast<int64_t>(raw_data.size()));
        auto device_scope = std::make_unique<DeviceScope>(device_);
        std::shared_ptr<T> data = std::make_unique<T[]>(shape.total_size());
        std::copy(raw_data.begin(), raw_data.end(), data.get());

        {
            Array x = Array::FromBuffer(shape, dtype, data);
            CheckCommonMembers(x, shape, dtype, device_);
            ExpectDataEqual<T>(data.get(), x);
            EXPECT_EQ(int64_t{sizeof(T)}, x.element_bytes());
            EXPECT_EQ(shape.total_size() * int64_t{sizeof(T)}, x.total_bytes());
        }
        {
            const Array x = Array::FromBuffer(shape, dtype, data);
            CheckCommonMembers(x, shape, dtype, device_);
            ExpectDataEqual<T>(data.get(), x);
            EXPECT_EQ(int64_t{sizeof(T)}, x.element_bytes());
            EXPECT_EQ(shape.total_size() * int64_t{sizeof(T)}, x.total_bytes());
        }
    }

    void CheckEmpty(const Shape& shape, Dtype dtype) {
        Array x = Array::Empty(shape, dtype, device_);
        CheckCommonMembers(x, shape, dtype, device_);
    }

    void CheckFullWithGivenDtype(const Shape& shape, Scalar scalar, Dtype dtype) {
        Array x = Array::Full(shape, scalar, dtype, device_);
        CheckCommonMembers(x, shape, dtype, device_);
        ExpectDataEqual(x, dtype, scalar);
    }

    void CheckFullWithScalarDtype(const Shape& shape, Scalar scalar) {
        Array x = Array::Full(shape, scalar, device_);
        CheckCommonMembers(x, shape, scalar.dtype(), device_);
        ExpectDataEqual(x, scalar.dtype(), scalar);
    }

    void CheckZeros(const Shape& shape, Dtype dtype) {
        Array x = Array::Zeros(shape, dtype, device_);
        CheckCommonMembers(x, shape, dtype, device_);
        ExpectDataEqual(x, dtype, 0);
    }

    void CheckOnes(const Shape& shape, Dtype dtype) {
        Array x = Array::Ones(shape, dtype, device_);
        CheckCommonMembers(x, shape, dtype, device_);
        ExpectDataEqual(x, dtype, 1);
    }

    void CheckEmptyLike(const Shape& shape, Dtype dtype) {
        Array x_orig = Array::Empty(shape, dtype, device_);
        Array x = Array::EmptyLike(x_orig, device_);
        CheckCommonLikeMembers(x, x_orig, device_);
    }

    void CheckFullLike(const Shape& shape, Scalar scalar, Dtype dtype) {
        Array x_orig = Array::Empty(shape, dtype, device_);
        Array x = Array::FullLike(x_orig, scalar, device_);
        CheckCommonLikeMembers(x, x_orig, device_);
        ExpectDataEqual(x, dtype, scalar);
    }

    void CheckZerosLike(const Shape& shape, Dtype dtype) {
        Array x_orig = Array::Empty(shape, dtype, device_);
        Array x = Array::ZerosLike(x_orig, device_);
        CheckCommonLikeMembers(x, x_orig, device_);
        ExpectDataEqual(x, dtype, 0);
    }

    void CheckOnesLike(const Shape& shape, Dtype dtype) {
        Array x_orig = Array::Empty(shape, dtype, device_);
        Array x = Array::OnesLike(x_orig, device_);
        CheckCommonLikeMembers(x, x_orig, device_);
        ExpectDataEqual(x, dtype, 1);
    }

private:
    std::unique_ptr<DeviceScope> device_scope_;
    Device device_;
};

TEST_P(ArrayDeviceTest, FromBuffer) { CheckFromBuffer({2, 3}, Dtype::kFloat32, {0.f, 1.f, 2.f, 3.f, 4.f, 5.f}); }
TEST_P(ArrayDeviceTest, Empty) { CheckEmpty({2, 3}, Dtype::kFloat32); }
TEST_P(ArrayDeviceTest, FullWithGivenDtype) { CheckFullWithGivenDtype({2, 3}, float{2.f}, Dtype::kFloat32); }
TEST_P(ArrayDeviceTest, FullWithScalarDtype) { CheckFullWithScalarDtype({2, 3}, float{2.f}); }
TEST_P(ArrayDeviceTest, Ones) { CheckOnes({2, 3}, Dtype::kFloat32); }
TEST_P(ArrayDeviceTest, Zeros) { CheckZeros({2, 3}, Dtype::kFloat32); }

TEST_P(ArrayDeviceTest, EmptyLike) { CheckEmptyLike({2, 3}, Dtype::kFloat32); }
TEST_P(ArrayDeviceTest, FullLike) { CheckFullLike({2, 3}, float{2.f}, Dtype::kFloat32); }
TEST_P(ArrayDeviceTest, ZerosLike) { CheckZerosLike({2, 3}, Dtype::kFloat32); }
TEST_P(ArrayDeviceTest, OnesLike) { CheckOnesLike({2, 3}, Dtype::kFloat32); }

#ifdef XCHAINER_ENABLE_CUDA
// Native and CUDA device combinations
const auto parameters = ::testing::Combine(::testing::Values(nonstd::nullopt, nonstd::optional<std::string>(std::string("cpu")),
                                                             nonstd::optional<std::string>(std::string("cuda"))),
                                           ::testing::Values(std::string("cpu"), std::string("cuda")));
#else
// Native device only
const auto parameters = ::testing::Combine(::testing::Values(nonstd::nullopt, nonstd::optional<std::string>(std::string("cpu"))),
                                           ::testing::Values(std::string("cpu")));
#endif  // XCHAINER_ENABLE_CUDA
INSTANTIATE_TEST_CASE_P(ForEachDevicePair, ArrayDeviceTest, parameters);

}  // namespace
}  // namespace xchainer
