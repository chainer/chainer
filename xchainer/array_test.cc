#include "xchainer/array.h"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <type_traits>

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA
#include <gtest/gtest.h>

#include "xchainer/array.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device.h"
#include "xchainer/op_node.h"

namespace xchainer {
namespace {

class ArrayTest : public ::testing::TestWithParam<::testing::tuple<std::string>> {
protected:
    virtual void SetUp() {
        std::string device_name = ::testing::get<0>(GetParam());
        device_scope_ = std::make_unique<DeviceScope>(device_name);
    }

    virtual void TearDown() { device_scope_.reset(); }

public:
    template <typename T>
    Array MakeArray(std::initializer_list<int64_t> shape, std::shared_ptr<void> data, bool requires_grad = false) {
        return {shape, TypeToDtype<T>, data, requires_grad};
    }

    template <typename T>
    Array MakeArray(std::initializer_list<int64_t> shape, std::initializer_list<T> data, bool requires_grad = false) {
        auto a = std::make_unique<T[]>(data.size());
        std::copy(data.begin(), data.end(), a.get());
        return {shape, TypeToDtype<T>, std::move(a), requires_grad};
    }

    template <typename T>
    void AssertEqual(const Array& expected, const Array& actual) {
        ASSERT_EQ(expected.dtype(), actual.dtype());
        ASSERT_EQ(expected.shape(), actual.shape());
        AssertDataEqual<T>(expected, actual);
    }

    template <typename T>
    void AssertDataEqual(const Array& expected, const Array& actual) {
#ifdef XCHAINER_ENABLE_CUDA
        std::string device_name = ::testing::get<0>(GetParam());
        if (device_name == "cuda") {
            cuda::CheckError(cudaDeviceSynchronize());
        }
#endif  // XCHAINER_ENABLE_CUDA
        auto total_size = expected.shape().total_size();
        const T* expected_data = static_cast<const T*>(expected.data().get());
        const T* actual_data = static_cast<const T*>(actual.data().get());
        for (decltype(total_size) i = 0; i < total_size; i++) {
            ASSERT_EQ(expected_data[i], actual_data[i]);
        }
    }

    template <typename T>
    void AssertDataEqual(T expected, const Array& actual) {
#ifdef XCHAINER_ENABLE_CUDA
        std::string device_name = ::testing::get<0>(GetParam());
        if (device_name == "cuda") {
            cuda::CheckError(cudaDeviceSynchronize());
        }
#endif  // XCHAINER_ENABLE_CUDA
        auto total_size = actual.shape().total_size();
        const T* actual_data = static_cast<const T*>(actual.data().get());
        for (decltype(total_size) i = 0; i < total_size; i++) {
            if (std::isnan(expected)) {
                ASSERT_TRUE(std::isnan(actual_data[i]));
            } else {
                ASSERT_EQ(expected, actual_data[i]);
            }
        }
    }

    bool IsPointerCudaManaged(const void* ptr) {
#ifdef XCHAINER_ENABLE_CUDA
        cudaPointerAttributes attr = {};
        cuda::CheckError(cudaPointerGetAttributes(&attr, ptr));
        return attr.isManaged != 0;
#else
        (void)ptr;
        return false;
#endif  // XCHAINER_ENABLE_CUDA
    }

    template <bool is_const>
    void CheckArray() {
        using TargetArray = std::conditional_t<is_const, const Array, Array>;

        std::shared_ptr<void> data = std::make_unique<float[]>(2 * 3 * 4);
        TargetArray x = MakeArray<float>({2, 3, 4}, data);

        // Basic attributes
        ASSERT_EQ(TypeToDtype<float>, x.dtype());
        ASSERT_EQ(3, x.ndim());
        ASSERT_EQ(2 * 3 * 4, x.total_size());
        ASSERT_EQ(4, x.element_bytes());
        ASSERT_EQ(2 * 3 * 4 * 4, x.total_bytes());
        ASSERT_EQ(0, x.offset());
        ASSERT_TRUE(x.is_contiguous());

        // Array::data
        std::shared_ptr<const void> x_data = x.data();
        if (GetCurrentDevice() == MakeDevice("cpu")) {
            ASSERT_EQ(data, x_data);
        } else if (GetCurrentDevice() == MakeDevice("cuda")) {
            ASSERT_NE(data, x_data);
            ASSERT_TRUE(IsPointerCudaManaged(x_data.get()));
        } else {
            FAIL() << "invalid device";
        }
    }

    template <typename T>
    void CheckEmpty() {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Empty(Shape{3, 2}, dtype);
        ASSERT_NE(x.data(), nullptr);
        ASSERT_EQ(x.shape(), Shape({3, 2}));
        ASSERT_EQ(x.dtype(), dtype);

        if (GetCurrentDevice() == MakeDevice("cpu")) {
            //
        } else if (GetCurrentDevice() == MakeDevice("cuda")) {
            ASSERT_TRUE(IsPointerCudaManaged(x.data().get()));
        } else {
            FAIL() << "invalid device";
        }
    }

    template <typename T>
    void CheckEmptyLike() {
        Dtype dtype = TypeToDtype<T>;
        Array x_orig = Array::Empty(Shape{3, 2}, dtype);
        Array x = Array::EmptyLike(x_orig);
        ASSERT_NE(x.data(), nullptr);
        ASSERT_NE(x.data(), x_orig.data());
        ASSERT_EQ(x.shape(), x_orig.shape());
        ASSERT_EQ(x.dtype(), x_orig.dtype());

        if (GetCurrentDevice() == MakeDevice("cpu")) {
            //
        } else if (GetCurrentDevice() == MakeDevice("cuda")) {
            ASSERT_TRUE(IsPointerCudaManaged(x.data().get()));
        } else {
            FAIL() << "invalid device";
        }
    }

    template <typename T>
    void CheckFill(T expected, Scalar scalar) {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Empty(Shape{3, 2}, dtype);
        x.Fill(scalar);
        AssertDataEqual(expected, x);
    }

    template <typename T>
    void CheckFill(T value) {
        CheckFill(value, value);
    }

    template <typename T>
    void CheckFullWithGivenDtype(T expected, Scalar scalar) {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Full(Shape{3, 2}, dtype, scalar);
        ASSERT_NE(x.data(), nullptr);
        ASSERT_EQ(x.shape(), Shape({3, 2}));
        ASSERT_EQ(x.dtype(), dtype);
        AssertDataEqual(expected, x);
    }

    template <typename T>
    void CheckFullWithGivenDtype(T value) {
        CheckFullWithGivenDtype(value, value);
    }

    template <typename T>
    void CheckFullWithScalarDtype(T value) {
        Scalar scalar = {value};
        Array x = Array::Full(Shape{3, 2}, scalar);
        ASSERT_NE(x.data(), nullptr);
        ASSERT_EQ(x.shape(), Shape({3, 2}));
        ASSERT_EQ(x.dtype(), scalar.dtype());
        AssertDataEqual(value, x);
    }

    template <typename T>
    void CheckFullLike(T expected, Scalar scalar) {
        Dtype dtype = TypeToDtype<T>;
        Array x_orig = Array::Empty(Shape{3, 2}, dtype);
        Array x = Array::FullLike(x_orig, scalar);
        ASSERT_NE(x.data(), nullptr);
        ASSERT_NE(x.data(), x_orig.data());
        ASSERT_EQ(x.shape(), x_orig.shape());
        ASSERT_EQ(x.dtype(), x_orig.dtype());
        AssertDataEqual(expected, x);
    }

    template <typename T>
    void CheckFullLike(T value) {
        CheckFullLike(value, value);
    }

    template <typename T>
    void CheckZeros() {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Zeros(Shape{3, 2}, dtype);
        ASSERT_NE(x.data(), nullptr);
        ASSERT_EQ(x.shape(), Shape({3, 2}));
        ASSERT_EQ(x.dtype(), dtype);
        T expected = static_cast<T>(0);
        AssertDataEqual(expected, x);
    }

    template <typename T>
    void CheckZerosLike() {
        Dtype dtype = TypeToDtype<T>;
        Array x_orig = Array::Empty(Shape{3, 2}, dtype);
        Array x = Array::ZerosLike(x_orig);
        ASSERT_NE(x.data(), nullptr);
        ASSERT_NE(x.data(), x_orig.data());
        ASSERT_EQ(x.shape(), x_orig.shape());
        ASSERT_EQ(x.dtype(), x_orig.dtype());
        T expected = static_cast<T>(0);
        AssertDataEqual(expected, x);
    }

    template <typename T>
    void CheckOnes() {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Ones(Shape{3, 2}, dtype);
        ASSERT_NE(x.data(), nullptr);
        ASSERT_EQ(x.shape(), Shape({3, 2}));
        ASSERT_EQ(x.dtype(), dtype);
        T expected = static_cast<T>(1);
        AssertDataEqual(expected, x);
    }

    template <typename T>
    void CheckOnesLike() {
        Dtype dtype = TypeToDtype<T>;
        Array x_orig = Array::Empty(Shape{3, 2}, dtype);
        Array x = Array::OnesLike(x_orig);
        ASSERT_NE(x.data(), nullptr);
        ASSERT_NE(x.data(), x_orig.data());
        ASSERT_EQ(x.shape(), x_orig.shape());
        ASSERT_EQ(x.dtype(), x_orig.dtype());
        T expected = static_cast<T>(1);
        AssertDataEqual(expected, x);
    }

private:
    std::unique_ptr<DeviceScope> device_scope_;
};

TEST_P(ArrayTest, ArrayCtor) { CheckArray<false>(); }

TEST_P(ArrayTest, ConstArrayCtor) { CheckArray<true>(); }

TEST_P(ArrayTest, SetRequiresGrad) {
    Array x = MakeArray<bool>({1}, {true});
    EXPECT_FALSE(x.requires_grad());
    x.set_requires_grad(true);
    EXPECT_TRUE(x.requires_grad());
    x.set_requires_grad(false);
    EXPECT_FALSE(x.requires_grad());
}

TEST_P(ArrayTest, Empty) {
    CheckEmpty<bool>();
    CheckEmpty<int8_t>();
    CheckEmpty<int16_t>();
    CheckEmpty<int32_t>();
    CheckEmpty<int64_t>();
    CheckEmpty<uint8_t>();
    CheckEmpty<float>();
    CheckEmpty<double>();
}

TEST_P(ArrayTest, EmptyLike) {
    CheckEmptyLike<bool>();
    CheckEmptyLike<int8_t>();
    CheckEmptyLike<int16_t>();
    CheckEmptyLike<int32_t>();
    CheckEmptyLike<int64_t>();
    CheckEmptyLike<uint8_t>();
    CheckEmptyLike<float>();
    CheckEmptyLike<double>();
}

TEST_P(ArrayTest, Fill) {
    CheckFill(true);
    CheckFill(false);
    CheckFill(static_cast<int8_t>(0));
    CheckFill(static_cast<int8_t>(-1));
    CheckFill(static_cast<int8_t>(5));
    CheckFill(static_cast<int8_t>(-128));
    CheckFill(static_cast<int8_t>(127));
    CheckFill(static_cast<int16_t>(0));
    CheckFill(static_cast<int16_t>(-3));
    CheckFill(static_cast<int32_t>(0));
    CheckFill(static_cast<int32_t>(-3));
    CheckFill(static_cast<int64_t>(0));
    CheckFill(static_cast<int64_t>(-3));
    CheckFill(static_cast<uint8_t>(0));
    CheckFill(static_cast<uint8_t>(255));
    CheckFill(static_cast<float>(0.f));
    CheckFill(static_cast<float>(std::numeric_limits<float>::infinity()));
    CheckFill(static_cast<float>(std::nanf("")));
    CheckFill(static_cast<double>(0.f));
    CheckFill(static_cast<double>(std::numeric_limits<double>::infinity()));
    CheckFill(static_cast<double>(std::nan("")));

    CheckFill(true, Scalar(1));
    CheckFill(true, Scalar(2));
    CheckFill(true, Scalar(-1));
    CheckFill(false, Scalar(0));
    CheckFill(static_cast<int8_t>(1), Scalar(1));
    CheckFill(static_cast<int8_t>(1), Scalar(1L));
    CheckFill(static_cast<int8_t>(1), Scalar(static_cast<uint8_t>(1)));
    CheckFill(static_cast<int8_t>(1), Scalar(true));
    CheckFill(static_cast<int8_t>(1), Scalar(1.0f));
    CheckFill(static_cast<int8_t>(1), Scalar(1.0));
    CheckFill(static_cast<int16_t>(1), Scalar(1));
    CheckFill(static_cast<int16_t>(1), Scalar(1L));
    CheckFill(static_cast<int16_t>(1), Scalar(static_cast<uint8_t>(1)));
    CheckFill(static_cast<int16_t>(1), Scalar(true));
    CheckFill(static_cast<int16_t>(1), Scalar(1.0f));
    CheckFill(static_cast<int16_t>(1), Scalar(1.0));
    CheckFill(static_cast<int32_t>(1), Scalar(1));
    CheckFill(static_cast<int32_t>(1), Scalar(1L));
    CheckFill(static_cast<int32_t>(1), Scalar(static_cast<uint8_t>(1)));
    CheckFill(static_cast<int32_t>(1), Scalar(true));
    CheckFill(static_cast<int32_t>(1), Scalar(1.0f));
    CheckFill(static_cast<int32_t>(1), Scalar(1.0));
    CheckFill(static_cast<int64_t>(1), Scalar(1));
    CheckFill(static_cast<int64_t>(1), Scalar(1L));
    CheckFill(static_cast<int64_t>(1), Scalar(static_cast<uint8_t>(1)));
    CheckFill(static_cast<int64_t>(1), Scalar(true));
    CheckFill(static_cast<int64_t>(1), Scalar(1.0f));
    CheckFill(static_cast<int64_t>(1), Scalar(1.0));
    CheckFill(static_cast<uint8_t>(1), Scalar(1));
    CheckFill(static_cast<uint8_t>(1), Scalar(1L));
    CheckFill(static_cast<uint8_t>(1), Scalar(static_cast<uint8_t>(1)));
    CheckFill(static_cast<uint8_t>(1), Scalar(true));
    CheckFill(static_cast<uint8_t>(1), Scalar(1.0f));
    CheckFill(static_cast<uint8_t>(1), Scalar(1.0));
    CheckFill(static_cast<float>(1), Scalar(1));
    CheckFill(static_cast<float>(1), Scalar(1L));
    CheckFill(static_cast<float>(1), Scalar(static_cast<uint8_t>(1)));
    CheckFill(static_cast<float>(1), Scalar(true));
    CheckFill(static_cast<float>(1), Scalar(1.0f));
    CheckFill(static_cast<float>(1), Scalar(1.0));
    CheckFill(static_cast<double>(1), Scalar(1));
    CheckFill(static_cast<double>(1), Scalar(1L));
    CheckFill(static_cast<double>(1), Scalar(static_cast<uint8_t>(1)));
    CheckFill(static_cast<double>(1), Scalar(true));
    CheckFill(static_cast<double>(1), Scalar(1.0f));
    CheckFill(static_cast<double>(1), Scalar(1.0));
}

TEST_P(ArrayTest, FullWithGivenDtype) {
    CheckFullWithGivenDtype(true);
    CheckFullWithGivenDtype(static_cast<int8_t>(2));
    CheckFullWithGivenDtype(static_cast<int16_t>(2));
    CheckFullWithGivenDtype(static_cast<int32_t>(2));
    CheckFullWithGivenDtype(static_cast<int64_t>(2));
    CheckFullWithGivenDtype(static_cast<uint8_t>(2));
    CheckFullWithGivenDtype(static_cast<float>(2.0f));
    CheckFullWithGivenDtype(static_cast<double>(2.0f));

    CheckFullWithGivenDtype(true, Scalar(1));
    CheckFullWithGivenDtype(true, Scalar(2));
    CheckFullWithGivenDtype(true, Scalar(-1));
    CheckFullWithGivenDtype(false, Scalar(0));
}

TEST_P(ArrayTest, FullWithScalarDtype) {
    CheckFullWithScalarDtype(true);
    CheckFullWithScalarDtype(static_cast<int8_t>(2));
    CheckFullWithScalarDtype(static_cast<int16_t>(2));
    CheckFullWithScalarDtype(static_cast<int32_t>(2));
    CheckFullWithScalarDtype(static_cast<int64_t>(2));
    CheckFullWithScalarDtype(static_cast<uint8_t>(2));
    CheckFullWithScalarDtype(static_cast<float>(2.0f));
    CheckFullWithScalarDtype(static_cast<double>(2.0f));
}

TEST_P(ArrayTest, FullLike) {
    CheckFullLike(true);
    CheckFullLike(static_cast<int8_t>(2));
    CheckFullLike(static_cast<int16_t>(2));
    CheckFullLike(static_cast<int32_t>(2));
    CheckFullLike(static_cast<int64_t>(2));
    CheckFullLike(static_cast<uint8_t>(2));
    CheckFullLike(static_cast<float>(2.0f));
    CheckFullLike(static_cast<double>(2.0f));

    CheckFullLike(true, Scalar(1));
    CheckFullLike(true, Scalar(2));
    CheckFullLike(true, Scalar(-1));
    CheckFullLike(false, Scalar(0));
}

TEST_P(ArrayTest, Zeros) {
    CheckZeros<bool>();
    CheckZeros<int8_t>();
    CheckZeros<int16_t>();
    CheckZeros<int32_t>();
    CheckZeros<int64_t>();
    CheckZeros<uint8_t>();
    CheckZeros<float>();
    CheckZeros<double>();
}

TEST_P(ArrayTest, ZerosLike) {
    CheckZerosLike<bool>();
    CheckZerosLike<int8_t>();
    CheckZerosLike<int16_t>();
    CheckZerosLike<int32_t>();
    CheckZerosLike<int64_t>();
    CheckZerosLike<uint8_t>();
    CheckZerosLike<float>();
    CheckZerosLike<double>();
}

TEST_P(ArrayTest, Ones) {
    CheckOnes<bool>();
    CheckOnes<int8_t>();
    CheckOnes<int16_t>();
    CheckOnes<int32_t>();
    CheckOnes<int64_t>();
    CheckOnes<uint8_t>();
    CheckOnes<float>();
    CheckOnes<double>();
}

TEST_P(ArrayTest, OnesLike) {
    CheckOnesLike<bool>();
    CheckOnesLike<int8_t>();
    CheckOnesLike<int16_t>();
    CheckOnesLike<int32_t>();
    CheckOnesLike<int64_t>();
    CheckOnesLike<uint8_t>();
    CheckOnesLike<float>();
    CheckOnesLike<double>();
}

TEST_P(ArrayTest, IAdd) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, true, true, false});
        a += b;
        AssertEqual<bool>(e, a);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {2, 4, 6});
        a += b;
        AssertEqual<int8_t>(e, a);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {2, 4, 6});
        a += b;
        AssertEqual<float>(e, a);
    }
}

TEST_P(ArrayTest, IMul) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, false, false, false});
        a *= b;
        AssertEqual<bool>(e, a);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {1, 4, 9});
        a *= b;
        AssertEqual<int8_t>(e, a);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {1, 4, 9});
        a *= b;
        AssertEqual<float>(e, a);
    }
}

TEST_P(ArrayTest, Add) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, true, true, false});
        Array o = a + b;
        AssertEqual<bool>(e, o);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {2, 4, 6});
        Array o = a + b;
        AssertEqual<int8_t>(e, o);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {2, 4, 6});
        Array o = a + b;
        AssertEqual<float>(e, o);
    }
}

TEST_P(ArrayTest, Mul) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, false, false, false});
        Array o = a * b;
        AssertEqual<bool>(e, o);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {1, 4, 9});
        Array o = a * b;
        AssertEqual<int8_t>(e, o);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {1, 4, 9});
        Array o = a * b;
        AssertEqual<float>(e, o);
    }
}

TEST_P(ArrayTest, ChainedMath) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array c = a * b;
        Array o = a + c;
        AssertEqual<bool>(e, o);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {2, 6, 12});
        Array c = a * b;
        Array o = a + c;
        AssertEqual<int8_t>(e, o);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {2, 6, 12});
        Array c = a * b;
        Array o = a + c;
        AssertEqual<float>(e, o);
    }
}

TEST_P(ArrayTest, ChainedInplaceMath) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, true, false, false});
        b *= a;
        a += b;
        AssertEqual<bool>(e, a);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {2, 6, 12});
        b *= a;
        a += b;
        AssertEqual<int8_t>(e, a);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {2, 6, 12});
        b *= a;
        a += b;
        AssertEqual<float>(e, a);
    }
}

TEST_P(ArrayTest, ComputationalGraph) {
    {
        // c = a + b
        // o = a * c
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false}, true);
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false}, true);
        {
            auto a_node = a.node();
            auto b_node = b.node();
            ASSERT_NE(a_node, nullptr);
            ASSERT_NE(b_node, nullptr);
            auto a_op_node = a_node->next_node();
            auto b_op_node = b_node->next_node();
            ASSERT_EQ(a_op_node, nullptr);
            ASSERT_EQ(b_op_node, nullptr);
        }

        Array c = a + b;
        {
            auto a_node = a.node();
            auto b_node = b.node();
            auto c_node = c.node();
            ASSERT_NE(a_node, nullptr);
            ASSERT_NE(b_node, nullptr);
            ASSERT_NE(c_node, nullptr);
            auto a_op_node = a_node->next_node();
            auto b_op_node = b_node->next_node();
            auto c_op_node = c_node->next_node();
            ASSERT_EQ(a_op_node, nullptr);
            ASSERT_EQ(b_op_node, nullptr);
            ASSERT_NE(c_op_node, nullptr);
            ASSERT_EQ(c_op_node->name(), "add");
        }

        Array o = a * c;
        {
            auto a_node = a.node();
            auto b_node = b.node();
            auto c_node = c.node();
            auto o_node = o.node();
            ASSERT_NE(a_node, nullptr);
            ASSERT_NE(b_node, nullptr);
            ASSERT_NE(c_node, nullptr);
            ASSERT_NE(o_node, nullptr);
            auto a_op_node = a_node->next_node();
            auto b_op_node = b_node->next_node();
            auto c_op_node = c_node->next_node();
            auto o_op_node = o_node->next_node();
            ASSERT_EQ(a_op_node, nullptr);
            ASSERT_EQ(b_op_node, nullptr);
            ASSERT_NE(c_op_node, nullptr);
            ASSERT_NE(o_op_node, nullptr);
            ASSERT_EQ(c_op_node->name(), "add");
            ASSERT_EQ(o_op_node->name(), "mul");
        }
    }
}

TEST_P(ArrayTest, InplaceNotAllowedWithRequiresGrad) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false}, true);
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false}, true);
        EXPECT_THROW({ a += b; }, XchainerError);
    }

    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false}, true);
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false}, true);
        EXPECT_THROW({ a *= b; }, XchainerError);
    }
}

TEST_P(ArrayTest, CopyCtor) {
    Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
    Array b = a;
    AssertEqual<bool>(a, b);

    // Deep copy, therefore assert different addresses to data
    ASSERT_NE(a.data().get(), b.data().get());
}

TEST_P(ArrayTest, AddBackward) {
    Array a = MakeArray<bool>({4, 1}, {true, true, false, false}, true);
    Array b = MakeArray<bool>({4, 1}, {true, false, true, false}, true);
    Array o = a + b;

    auto op_node = o.node()->next_node();
    Array go = MakeArray<bool>({4, 1}, {true, true, true, true});
    Array ga = op_node->backward_functions()[0](go);
    Array gb = op_node->backward_functions()[1](go);

    AssertEqual<bool>(ga, go);
    AssertEqual<bool>(gb, go);
}

TEST_P(ArrayTest, MulBackward) {
    Array a = MakeArray<bool>({4, 1}, {true, true, false, false}, true);
    Array b = MakeArray<bool>({4, 1}, {true, false, true, false}, true);
    Array o = a * b;

    auto op_node = o.node()->next_node();
    Array go = MakeArray<bool>({4, 1}, {true, true, true, true});
    Array ga = op_node->backward_functions()[0](go);
    Array gb = op_node->backward_functions()[1](go);

    AssertEqual<bool>(ga, go * b);
    AssertEqual<bool>(gb, go * a);
}

INSTANTIATE_TEST_CASE_P(ForEachDevice, ArrayTest, ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                                                      std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                                                      std::string{"cpu"}));

}  // namespace
}  // namespace xchainer
