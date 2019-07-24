#include "chainerx/testing/array.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <absl/types/optional.h>
#include <gtest/gtest.h>

#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/macro.h"
#include "chainerx/shape.h"
#include "chainerx/strides.h"
#include "chainerx/testing/array_check.h"
#include "chainerx/testing/device_session.h"

namespace chainerx {
namespace testing {
namespace {

class TestingArrayTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        const std::string& backend_name = GetParam();
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

    Device& device() {
        CHAINERX_ASSERT(device_session_.has_value());
        return device_session_->device();
    }

private:
    absl::optional<DeviceSession> device_session_;
};

template <typename T>
void ExpectArrayAttr(
        const Shape& expected_shape,
        Dtype expected_dtype,
        const Strides& expected_strides,
        const std::vector<T>& expected_data,
        Device& expected_device,
        const Array& actual) {
    EXPECT_EQ(expected_shape, actual.shape());
    EXPECT_EQ(expected_dtype, actual.dtype());
    EXPECT_EQ(expected_strides, actual.strides());
    EXPECT_EQ(&expected_device, &actual.device());
    ExpectDataEqual<T>(expected_data, actual);
}

TEST_P(TestingArrayTest, BuildArrayBuildMethods) {
    using T = float;
    Shape shape{2, 3};
    Dtype dtype = TypeToDtype<T>;
    std::vector<T> data{1, 2, 3, 4, 5, 6};
    Strides strides{3 * sizeof(T), sizeof(T)};

    {
        Array a = BuildArray(shape).WithData<T>(data).Build();
        ExpectArrayAttr(shape, dtype, strides, data, device(), a);
    }
    {
        Array a = *BuildArray(shape).WithData<T>(data);
        ExpectArrayAttr(shape, dtype, strides, data, device(), a);
    }
    {
        Array a = static_cast<Array>(BuildArray(shape).WithData<T>(data));
        ExpectArrayAttr(shape, dtype, strides, data, device(), a);
    }
}

TEST_P(TestingArrayTest, BuildArrayZeroDimensional) {
    using T = float;
    Shape shape{};
    Dtype dtype = TypeToDtype<T>;
    std::vector<T> data{1};
    Strides strides{};

    Array a = BuildArray(shape).WithData<T>(data);
    ExpectArrayAttr(shape, dtype, strides, data, device(), a);
}

TEST_P(TestingArrayTest, BuildArrayOneDimensional) {
    using T = float;
    Shape shape{1};
    Dtype dtype = TypeToDtype<T>;
    std::vector<T> data{1};
    Strides strides{sizeof(T)};

    Array a = BuildArray(shape).WithData<T>(data);
    ExpectArrayAttr(shape, dtype, strides, data, device(), a);
}

TEST_P(TestingArrayTest, BuildArrayMultiDimensional) {
    using T = float;
    Shape shape{1, 2, 3};
    Dtype dtype = TypeToDtype<T>;
    std::vector<T> data{1, 2, 3, 4, 5, 6};
    Strides strides{6 * sizeof(T), 3 * sizeof(T), sizeof(T)};

    Array a = BuildArray(shape).WithData<T>(data);
    ExpectArrayAttr(shape, dtype, strides, data, device(), a);
}

TEST_P(TestingArrayTest, BuildArrayMultiDimensionalFromDataIterators) {
    using T = float;
    Shape shape{1, 2, 3};
    Dtype dtype = TypeToDtype<T>;
    std::vector<T> data{1, 2, 3, 4, 5, 6};
    Strides strides{6 * sizeof(T), 3 * sizeof(T), sizeof(T)};

    Array a = BuildArray(shape).WithData<T>(data.begin(), data.end());
    ExpectArrayAttr(shape, dtype, strides, data, device(), a);
}

TEST_P(TestingArrayTest, BuildArrayLinearDataDefaultStep) {
    using T = float;
    Shape shape{2, 3};
    Dtype dtype = TypeToDtype<T>;
    Strides strides{3 * sizeof(T), sizeof(T)};

    Array a = BuildArray(shape).WithLinearData<T>(2);
    ExpectArrayAttr(shape, dtype, strides, std::vector<T>{2, 3, 4, 5, 6, 7}, device(), a);
}

TEST_P(TestingArrayTest, BuildArrayLinearDataExplicitStep) {
    using T = float;
    Shape shape{2, 3};
    Dtype dtype = TypeToDtype<T>;
    Strides strides{3 * sizeof(T), sizeof(T)};

    // Positive step
    {
        Array a = BuildArray(shape).WithLinearData<T>(2, 3);
        ExpectArrayAttr(shape, dtype, strides, std::vector<T>{2, 5, 8, 11, 14, 17}, device(), a);
    }
    // Negative step
    {
        Array a = BuildArray(shape).WithLinearData<T>(3, -1);
        ExpectArrayAttr(shape, dtype, strides, std::vector<T>{3, 2, 1, 0, -1, -2}, device(), a);
    }
}

TEST_P(TestingArrayTest, BuildArraySinglePaddingForAllDimensions) {
    using T = float;
    Shape shape{1, 2, 3};
    Dtype dtype = TypeToDtype<T>;
    std::vector<T> data{1, 2, 3, 4, 5, 6};
    int64_t padding = 2;
    int64_t item_size = sizeof(T);  // Cast to int64_t to avoid implicit narrowing.
    Strides strides{((item_size + padding * item_size) * shape[2] + padding * item_size) * shape[1] + padding * item_size,
                    (item_size + padding * item_size) * shape[2] + padding * item_size,
                    item_size + padding * item_size};

    Array a = BuildArray(shape).WithData<T>(data).WithPadding(padding);
    ExpectArrayAttr(shape, dtype, strides, data, device(), a);
}

TEST_P(TestingArrayTest, BuildArrayPaddingPerDimension) {
    using T = float;
    Shape shape{1, 2, 3};
    Dtype dtype = TypeToDtype<T>;
    std::vector<T> data{1, 2, 3, 4, 5, 6};
    std::vector<int64_t> padding{2, 3, 1};
    int64_t item_size = sizeof(T);  // Cast to int64_t to avoid implicit narrowing.
    Strides strides{((item_size + padding[2] * item_size) * shape[2] + padding[1] * item_size) * shape[1] + padding[0] * item_size,
                    (item_size + padding[2] * item_size) * shape[2] + padding[1] * item_size,
                    item_size + padding[2] * item_size};

    Array a = BuildArray(shape).WithData<T>(data).WithPadding(padding);
    ExpectArrayAttr(shape, dtype, strides, data, device(), a);
}

TEST_P(TestingArrayTest, BuildArrayNegativePadding) {
    using T = float;
    Shape shape{2, 3};
    Dtype dtype = TypeToDtype<T>;
    std::vector<T> data{1, 2, 3, 4, 5, 6};
    std::vector<int64_t> padding{-3, 1};
    int64_t item_size = sizeof(T);  // Cast to int64_t to avoid implicit narrowing.
    Strides strides{(item_size + padding[1] * item_size) * shape[1] + padding[0] * item_size, item_size + padding[1] * item_size};

    Array a = BuildArray(shape).WithData<T>(data).WithPadding(padding);
    ExpectArrayAttr(shape, dtype, strides, data, device(), a);
}

TEST_P(TestingArrayTest, BuildArrayDevice) {
    using T = float;
    Shape shape{};
    Dtype dtype = TypeToDtype<T>;
    std::vector<T> data{1};
    Strides strides{};

    Array a = BuildArray(shape).WithData<T>(data).WithDevice(device());
    ExpectArrayAttr(shape, dtype, strides, data, device(), a);
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        TestingArrayTest,
        ::testing::Values(
#ifdef CHAINERX_ENABLE_CUDA
                std::string{"cuda"},
#endif  // CHAINERX_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace testing
}  // namespace chainerx
