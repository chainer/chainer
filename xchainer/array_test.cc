#include "xchainer/array.h"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <string>
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
#include "xchainer/error.h"
#include "xchainer/memory.h"
#include "xchainer/op_node.h"

namespace xchainer {
namespace {

class ArrayTest : public ::testing::TestWithParam<::testing::tuple<std::string>> {
protected:
    void SetUp() override {
        std::string device_name = ::testing::get<0>(GetParam());
        device_scope_ = std::make_unique<DeviceScope>(device_name);
    }

    void TearDown() override { device_scope_.reset(); }

public:
    template <typename T>
    Array MakeArray(const Shape& shape, std::shared_ptr<void> data, bool requires_grad = false, const std::string& graph_name = "") {
        Array arr = Array::FromBuffer(shape, TypeToDtype<T>, data);
        arr.GetNode(graph_name, requires_grad);
        return arr;
    }

    template <typename T>
    Array MakeArray(const Shape& shape, std::initializer_list<T> data, bool requires_grad = false, const std::string& graph_name = "") {
        auto a = std::make_unique<T[]>(data.size());
        std::copy(data.begin(), data.end(), a.get());
        return MakeArray<T>(shape, std::move(a), requires_grad, graph_name);
    }

    template <typename T>
    void ExpectEqualCopy(const Array& expected, const Array& actual, const std::string& graph_name = "") {
        EXPECT_EQ(expected.dtype(), actual.dtype());
        EXPECT_EQ(expected.shape(), actual.shape());

        // Deep copy, therefore assert different addresses to data
        EXPECT_NE(expected.data().get(), actual.data().get());

        EXPECT_EQ(expected.requires_grad(graph_name), actual.requires_grad(graph_name));
        EXPECT_EQ(expected.requires_grad(), actual.requires_grad());
        EXPECT_TRUE(actual.is_contiguous());
        EXPECT_EQ(0, actual.offset());

        ExpectDataEqual<T>(expected, actual);
    }

    template <typename T>
    void ExpectEqual(const Array& expected, const Array& actual) {
        EXPECT_EQ(expected.dtype(), actual.dtype());
        EXPECT_EQ(expected.shape(), actual.shape());
        ExpectDataEqual<T>(expected, actual);
    }

    template <typename T>
    void ExpectDataEqual(const Array& expected, const Array& actual) {
        const T* expected_data = static_cast<const T*>(expected.data().get());
        ExpectDataEqual(expected_data, actual);
    }

    template <typename T>
    void ExpectDataEqual(const T* expected_data, const Array& actual) {
#ifdef XCHAINER_ENABLE_CUDA
        std::string device_name = ::testing::get<0>(GetParam());
        if (device_name == "cuda") {
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
        std::string device_name = ::testing::get<0>(GetParam());
        if (device_name == "cuda") {
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

    void ExpectDataExistsOnCurrentDevice(const Array& array) {
        if (GetCurrentDevice() == MakeDevice("cpu")) {
            EXPECT_FALSE(internal::IsPointerCudaMemory(array.data().get()));
        } else if (GetCurrentDevice() == MakeDevice("cuda")) {
            EXPECT_TRUE(internal::IsPointerCudaMemory(array.data().get()));
        } else {
            FAIL() << "invalid device";
        }
    }

    template <bool is_const, typename T>
    void CheckFromBuffer(const Shape& shape, std::initializer_list<T> raw_data) {
        using TargetArray = std::conditional_t<is_const, const Array, Array>;

        // Check test data
        ASSERT_EQ(shape.total_size(), static_cast<int64_t>(raw_data.size()));

        std::shared_ptr<T> data = std::make_unique<T[]>(shape.total_size());
        std::copy(raw_data.begin(), raw_data.end(), data.get());

        Dtype dtype = TypeToDtype<T>;
        TargetArray x = Array::FromBuffer(shape, dtype, data);

        // Basic attributes
        EXPECT_EQ(shape, x.shape());
        EXPECT_EQ(dtype, x.dtype());
        EXPECT_EQ(2, x.ndim());
        EXPECT_EQ(3 * 2, x.total_size());
        EXPECT_EQ(int64_t{sizeof(T)}, x.element_bytes());
        EXPECT_EQ(shape.total_size() * sizeof(T), x.total_bytes());
        EXPECT_TRUE(x.is_contiguous());
        EXPECT_EQ(0, x.offset());

        // Array::data
        ExpectDataEqual<T>(data.get(), x);
        ExpectDataExistsOnCurrentDevice(x);
        if (GetCurrentDevice() == MakeDevice("cpu")) {
            EXPECT_EQ(data.get(), x.data().get());
        } else if (GetCurrentDevice() == MakeDevice("cuda")) {
            EXPECT_NE(data.get(), x.data().get());
        } else {
            FAIL() << "invalid device";
        }
    }

    template <typename T>
    void CheckEmpty() {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Empty(Shape{3, 2}, dtype);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_EQ(x.shape(), Shape({3, 2}));
        EXPECT_EQ(x.dtype(), dtype);
        EXPECT_TRUE(x.is_contiguous());
        EXPECT_EQ(0, x.offset());
        ExpectDataExistsOnCurrentDevice(x);
    }

    template <typename T>
    void CheckEmptyLike() {
        Dtype dtype = TypeToDtype<T>;
        Array x_orig = Array::Empty(Shape{3, 2}, dtype);
        Array x = Array::EmptyLike(x_orig);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_NE(x.data(), x_orig.data());
        EXPECT_EQ(x.shape(), x_orig.shape());
        EXPECT_EQ(x.dtype(), x_orig.dtype());
        EXPECT_TRUE(x.is_contiguous());
        EXPECT_EQ(0, x.offset());
        ExpectDataExistsOnCurrentDevice(x);
    }

    template <typename T>
    void CheckFill(T expected, Scalar scalar) {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Empty(Shape{3, 2}, dtype);
        x.Fill(scalar);
        ExpectDataEqual(expected, x);
    }

    template <typename T>
    void CheckFill(T value) {
        CheckFill(value, value);
    }

    template <typename T>
    void CheckFullWithGivenDtype(T expected, Scalar scalar) {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Full(Shape{3, 2}, scalar, dtype);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_EQ(x.shape(), Shape({3, 2}));
        EXPECT_EQ(x.dtype(), dtype);
        EXPECT_TRUE(x.is_contiguous());
        EXPECT_EQ(0, x.offset());
        ExpectDataEqual(expected, x);
        ExpectDataExistsOnCurrentDevice(x);
    }

    template <typename T>
    void CheckFullWithGivenDtype(T value) {
        CheckFullWithGivenDtype(value, value);
    }

    template <typename T>
    void CheckFullWithScalarDtype(T value) {
        Scalar scalar = {value};
        Array x = Array::Full(Shape{3, 2}, scalar);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_EQ(x.shape(), Shape({3, 2}));
        EXPECT_EQ(x.dtype(), scalar.dtype());
        EXPECT_TRUE(x.is_contiguous());
        EXPECT_EQ(0, x.offset());
        ExpectDataEqual(value, x);
        ExpectDataExistsOnCurrentDevice(x);
    }

    template <typename T>
    void CheckFullLike(T expected, Scalar scalar) {
        Dtype dtype = TypeToDtype<T>;
        Array x_orig = Array::Empty(Shape{3, 2}, dtype);
        Array x = Array::FullLike(x_orig, scalar);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_NE(x.data(), x_orig.data());
        EXPECT_EQ(x.shape(), x_orig.shape());
        EXPECT_EQ(x.dtype(), x_orig.dtype());
        EXPECT_TRUE(x.is_contiguous());
        EXPECT_EQ(0, x.offset());
        ExpectDataEqual(expected, x);
        ExpectDataExistsOnCurrentDevice(x);
    }

    template <typename T>
    void CheckFullLike(T value) {
        CheckFullLike(value, value);
    }

    template <typename T>
    void CheckZeros() {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Zeros(Shape{3, 2}, dtype);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_EQ(x.shape(), Shape({3, 2}));
        EXPECT_EQ(x.dtype(), dtype);
        EXPECT_TRUE(x.is_contiguous());
        EXPECT_EQ(0, x.offset());
        T expected{0};
        ExpectDataEqual(expected, x);
        ExpectDataExistsOnCurrentDevice(x);
    }

    template <typename T>
    void CheckZerosLike() {
        Dtype dtype = TypeToDtype<T>;
        Array x_orig = Array::Empty(Shape{3, 2}, dtype);
        Array x = Array::ZerosLike(x_orig);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_NE(x.data(), x_orig.data());
        EXPECT_EQ(x.shape(), x_orig.shape());
        EXPECT_EQ(x.dtype(), x_orig.dtype());
        EXPECT_TRUE(x.is_contiguous());
        EXPECT_EQ(0, x.offset());
        T expected{0};
        ExpectDataEqual(expected, x);
        ExpectDataExistsOnCurrentDevice(x);
    }

    template <typename T>
    void CheckOnes() {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Ones(Shape{3, 2}, dtype);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_EQ(x.shape(), Shape({3, 2}));
        EXPECT_EQ(x.dtype(), dtype);
        EXPECT_TRUE(x.is_contiguous());
        EXPECT_EQ(0, x.offset());
        T expected{1};
        ExpectDataEqual(expected, x);
        ExpectDataExistsOnCurrentDevice(x);
    }

    template <typename T>
    void CheckOnesLike() {
        Dtype dtype = TypeToDtype<T>;
        Array x_orig = Array::Empty(Shape{3, 2}, dtype);
        Array x = Array::OnesLike(x_orig);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_NE(x.data(), x_orig.data());
        EXPECT_EQ(x.shape(), x_orig.shape());
        EXPECT_EQ(x.dtype(), x_orig.dtype());
        EXPECT_TRUE(x.is_contiguous());
        EXPECT_EQ(0, x.offset());
        T expected{1};
        ExpectDataEqual(expected, x);
        ExpectDataExistsOnCurrentDevice(x);
    }

private:
    std::unique_ptr<DeviceScope> device_scope_;
};

/*
TEST_P(ArrayTest, Temporary) {
    Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
    a.GetNode("");
    a.GetNode("graph_1");
    EXPECT_THROW(a.GetNode("graph_1"), XchainerError);
    // Array b = a;
    // ExpectEqualCopy<bool>(a, b);
}

*/
TEST_P(ArrayTest, CopyCtor) {
    Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
    Array b = a;
    ExpectEqualCopy<bool>(a, b);
}

TEST_P(ArrayTest, ArrayMoveCtor) {
    { EXPECT_TRUE(std::is_nothrow_move_constructible<Array>::value); }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        auto body = a.body();
        Array b = a;
        Array c = std::move(a);
        EXPECT_EQ(a.body(), nullptr);
        EXPECT_EQ(body, c.body());
        ExpectEqualCopy<float>(b, c);
    }
}

TEST_P(ArrayTest, ArrayBodyCtor) {
    Array a = MakeArray<float>({3, 1}, {1, 2, 3});
    auto body = a.body();
    Array b{body};
    EXPECT_EQ(body, b.body());

    EXPECT_EQ(a.dtype(), b.dtype());
    EXPECT_EQ(a.shape(), b.shape());
    EXPECT_EQ(a.data(), b.data());
    EXPECT_EQ(a.requires_grad(), b.requires_grad());
    EXPECT_EQ(a.is_contiguous(), b.is_contiguous());
    EXPECT_EQ(a.offset(), b.offset());
    EXPECT_EQ(a.node(), b.node());
}

TEST_P(ArrayTest, ArrayMoveAssignmentOperator) {
    {
        // TODO(hvy): Change the following expectations when copy assignment is implemented (not explicitly deleted)
        EXPECT_FALSE(std::is_nothrow_move_assignable<Array>::value);
    }
}

TEST_P(ArrayTest, SetRequiresGrad) {
    Array x = MakeArray<bool>({1}, {true}, false);
    ASSERT_FALSE(x.requires_grad());
    x.set_requires_grad(true);
    ASSERT_TRUE(x.requires_grad());
    x.set_requires_grad(false);
    ASSERT_FALSE(x.requires_grad());
}

TEST_P(ArrayTest, Grad) {
    Shape shape{2, 3};
    using T = float;

    Array x = MakeArray<T>(shape, {5, 3, 2, 1, 4, 6});
    Array g = MakeArray<T>(shape, {8, 4, 6, 3, 2, 1});

    EXPECT_FALSE(x.grad()) << "grad must be initially unset";

    // Set and get grad
    {
        x.set_grad(g);

        ASSERT_TRUE(x.grad());
        ExpectEqual<T>(g, *x.grad());
    }

    // Get grad multiple times
    {
        const nonstd::optional<Array>& grad1 = x.grad();
        const nonstd::optional<Array>& grad2 = x.grad();
        EXPECT_EQ(&*grad1, &*grad2) << "Multiple retrieval of grad must return the same arrays";
    }

    // ClearGrad
    {
        Array grad_view = x.grad()->MakeView();  // Make a view of grad

        x.ClearGrad();

        EXPECT_FALSE(x.grad()) << "grad must be cleared after calling ClearGrad()";

        // ClearGrad() must not affect previously retrieved view to grad
        ExpectEqual<T>(grad_view, g);
    }
}

TEST_P(ArrayTest, ArrayFromBuffer) {
    Shape shape{3, 2};
    CheckFromBuffer<false, bool>(shape, {true, false, false, true, false, true});
    CheckFromBuffer<false, int8_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<false, int16_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<false, int32_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<false, int64_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<false, uint8_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<false, float>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<false, double>(shape, {0, 1, 2, 3, 4, 5});
}

TEST_P(ArrayTest, ConstArrayFromBuffer) {
    Shape shape{3, 2};
    CheckFromBuffer<true, bool>(shape, {true, false, false, true, false, true});
    CheckFromBuffer<true, int8_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<true, int16_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<true, int32_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<true, int64_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<true, uint8_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<true, float>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<true, double>(shape, {0, 1, 2, 3, 4, 5});
}

#ifdef XCHAINER_ENABLE_CUDA
TEST_P(ArrayTest, FromBufferFromNonManagedMemory) {
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
    CheckFill(int8_t{0});
    CheckFill(int8_t{-1});
    CheckFill(int8_t{5});
    CheckFill(int8_t{-128});
    CheckFill(int8_t{127});
    CheckFill(int16_t{0});
    CheckFill(int16_t{-3});
    CheckFill(int32_t{0});
    CheckFill(int32_t{-3});
    CheckFill(int64_t{0});
    CheckFill(int64_t{-3});
    CheckFill(uint8_t{0});
    CheckFill(uint8_t{255});
    CheckFill(float{0});
    CheckFill(float{std::numeric_limits<float>::infinity()});
    CheckFill(float{std::nanf("")});
    CheckFill(double{0});
    CheckFill(double{std::numeric_limits<double>::infinity()});
    CheckFill(double{std::nan("")});

    CheckFill(true, Scalar(int32_t{1}));
    CheckFill(true, Scalar(int32_t{2}));
    CheckFill(true, Scalar(int32_t{-1}));
    CheckFill(false, Scalar(int32_t{0}));
    CheckFill(int8_t{1}, Scalar(int32_t{1}));
    CheckFill(int8_t{1}, Scalar(int64_t{1}));
    CheckFill(int8_t{1}, Scalar(uint8_t{1}));
    CheckFill(int8_t{1}, Scalar(true));
    CheckFill(int8_t{1}, Scalar(1.0f));
    CheckFill(int8_t{1}, Scalar(1.0));
    CheckFill(int16_t{1}, Scalar(int32_t{1}));
    CheckFill(int16_t{1}, Scalar(int64_t{1}));
    CheckFill(int16_t{1}, Scalar(uint8_t{1}));
    CheckFill(int16_t{1}, Scalar(true));
    CheckFill(int16_t{1}, Scalar(1.0f));
    CheckFill(int16_t{1}, Scalar(1.0));
    CheckFill(int32_t{1}, Scalar(int32_t{1}));
    CheckFill(int32_t{1}, Scalar(int64_t{1}));
    CheckFill(int32_t{1}, Scalar(uint8_t{1}));
    CheckFill(int32_t{1}, Scalar(true));
    CheckFill(int32_t{1}, Scalar(1.0f));
    CheckFill(int32_t{1}, Scalar(1.0));
    CheckFill(int64_t{1}, Scalar(int32_t{1}));
    CheckFill(int64_t{1}, Scalar(int64_t{1}));
    CheckFill(int64_t{1}, Scalar(uint8_t{1}));
    CheckFill(int64_t{1}, Scalar(true));
    CheckFill(int64_t{1}, Scalar(1.0f));
    CheckFill(int64_t{1}, Scalar(1.0));
    CheckFill(uint8_t{1}, Scalar(int32_t{1}));
    CheckFill(uint8_t{1}, Scalar(int64_t{1}));
    CheckFill(uint8_t{1}, Scalar(uint8_t{1}));
    CheckFill(uint8_t{1}, Scalar(true));
    CheckFill(uint8_t{1}, Scalar(1.0f));
    CheckFill(uint8_t{1}, Scalar(1.0));
    CheckFill(float{1}, Scalar(int32_t{1}));
    CheckFill(float{1}, Scalar(int64_t{1}));
    CheckFill(float{1}, Scalar(uint8_t{1}));
    CheckFill(float{1}, Scalar(true));
    CheckFill(float{1}, Scalar(1.0f));
    CheckFill(float{1}, Scalar(1.0));
    CheckFill(double{1}, Scalar(int32_t{1}));
    CheckFill(double{1}, Scalar(int64_t{1}));
    CheckFill(double{1}, Scalar(uint8_t{1}));
    CheckFill(double{1}, Scalar(true));
    CheckFill(double{1}, Scalar(1.0f));
    CheckFill(double{1}, Scalar(1.0));
}

TEST_P(ArrayTest, FullWithGivenDtype) {
    CheckFullWithGivenDtype(true);
    CheckFullWithGivenDtype(int8_t{2});
    CheckFullWithGivenDtype(int16_t{2});
    CheckFullWithGivenDtype(int32_t{2});
    CheckFullWithGivenDtype(int64_t{2});
    CheckFullWithGivenDtype(uint8_t{2});
    CheckFullWithGivenDtype(float{2.0f});
    CheckFullWithGivenDtype(double{2.0});

    CheckFullWithGivenDtype(true, Scalar(int32_t{1}));
    CheckFullWithGivenDtype(true, Scalar(int32_t{2}));
    CheckFullWithGivenDtype(true, Scalar(int32_t{-1}));
    CheckFullWithGivenDtype(false, Scalar(int32_t{0}));
}

TEST_P(ArrayTest, FullWithScalarDtype) {
    CheckFullWithScalarDtype(true);
    CheckFullWithScalarDtype(int8_t{2});
    CheckFullWithScalarDtype(int16_t{2});
    CheckFullWithScalarDtype(int32_t{2});
    CheckFullWithScalarDtype(int64_t{2});
    CheckFullWithScalarDtype(uint8_t{2});
    CheckFullWithScalarDtype(float{2.0f});
    CheckFullWithScalarDtype(double{2.0});
}

TEST_P(ArrayTest, FullLike) {
    CheckFullLike(true);
    CheckFullLike(int8_t{2});
    CheckFullLike(int16_t{2});
    CheckFullLike(int32_t{2});
    CheckFullLike(int64_t{2});
    CheckFullLike(uint8_t{2});
    CheckFullLike(float{2.0f});
    CheckFullLike(double{2.0});

    CheckFullLike(true, Scalar(int32_t{1}));
    CheckFullLike(true, Scalar(int32_t{2}));
    CheckFullLike(true, Scalar(int32_t{-1}));
    CheckFullLike(false, Scalar(int32_t{0}));
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
        ExpectEqual<bool>(e, a);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {2, 4, 6});
        a += b;
        ExpectEqual<int8_t>(e, a);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {2, 4, 6});
        a += b;
        ExpectEqual<float>(e, a);
    }
}

TEST_P(ArrayTest, IMul) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, false, false, false});
        a *= b;
        ExpectEqual<bool>(e, a);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {1, 4, 9});
        a *= b;
        ExpectEqual<int8_t>(e, a);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {1, 4, 9});
        a *= b;
        ExpectEqual<float>(e, a);
    }
}

TEST_P(ArrayTest, Add) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, true, true, false});
        Array o = a + b;
        ExpectEqual<bool>(e, o);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {2, 4, 6});
        Array o = a + b;
        ExpectEqual<int8_t>(e, o);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {2, 4, 6});
        Array o = a + b;
        ExpectEqual<float>(e, o);
    }
}

TEST_P(ArrayTest, Mul) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, false, false, false});
        Array o = a * b;
        ExpectEqual<bool>(e, o);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {1, 4, 9});
        Array o = a * b;
        ExpectEqual<int8_t>(e, o);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {1, 4, 9});
        Array o = a * b;
        ExpectEqual<float>(e, o);
    }
}

TEST_P(ArrayTest, ChainedMath) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array c = a * b;
        Array o = a + c;
        ExpectEqual<bool>(e, o);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {2, 6, 12});
        Array c = a * b;
        Array o = a + c;
        ExpectEqual<int8_t>(e, o);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {2, 6, 12});
        Array c = a * b;
        Array o = a + c;
        ExpectEqual<float>(e, o);
    }
}

TEST_P(ArrayTest, ChainedInplaceMath) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, true, false, false});
        b *= a;
        a += b;
        ExpectEqual<bool>(e, a);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {2, 6, 12});
        b *= a;
        a += b;
        ExpectEqual<int8_t>(e, a);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {2, 6, 12});
        b *= a;
        a += b;
        ExpectEqual<float>(e, a);
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
            EXPECT_NE(a_node, nullptr);
            EXPECT_NE(b_node, nullptr);
            auto a_op_node = a_node->next_node();
            auto b_op_node = b_node->next_node();
            EXPECT_EQ(a_op_node, nullptr);
            EXPECT_EQ(b_op_node, nullptr);
        }

        Array c = a + b;
        {
            auto a_node = a.node();
            auto b_node = b.node();
            auto c_node = c.node();
            EXPECT_NE(a_node, nullptr);
            EXPECT_NE(b_node, nullptr);
            EXPECT_NE(c_node, nullptr);
            auto a_op_node = a_node->next_node();
            auto b_op_node = b_node->next_node();
            auto c_op_node = c_node->next_node();
            EXPECT_EQ(a_op_node, nullptr);
            EXPECT_EQ(b_op_node, nullptr);
            EXPECT_NE(c_op_node, nullptr);
            EXPECT_EQ(c_op_node->name(), "add");
        }

        Array o = a * c;
        {
            auto a_node = a.node();
            auto b_node = b.node();
            auto c_node = c.node();
            auto o_node = o.node();
            EXPECT_NE(a_node, nullptr);
            EXPECT_NE(b_node, nullptr);
            EXPECT_NE(c_node, nullptr);
            EXPECT_NE(o_node, nullptr);
            auto a_op_node = a_node->next_node();
            auto b_op_node = b_node->next_node();
            auto c_op_node = c_node->next_node();
            auto o_op_node = o_node->next_node();
            EXPECT_EQ(a_op_node, nullptr);
            EXPECT_EQ(b_op_node, nullptr);
            EXPECT_NE(c_op_node, nullptr);
            EXPECT_NE(o_op_node, nullptr);
            EXPECT_EQ(c_op_node->name(), "add");
            EXPECT_EQ(o_op_node->name(), "mul");
        }
    }
}

/*
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
*/

TEST_P(ArrayTest, Copy) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array o = a.Copy();
        ExpectEqualCopy<bool>(a, o);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array o = a.Copy();
        ExpectEqualCopy<bool>(a, o);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array o = a.Copy();
        ExpectEqualCopy<bool>(a, o);
    }
}

TEST_P(ArrayTest, MakeView) {
    Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
    Array b = a.MakeView();
    ExpectEqual<bool>(a, b);
    // View should point same addresses to data
    EXPECT_EQ(a.data().get(), b.data().get());
}

TEST_P(ArrayTest, AddBackward) {
    Array a = MakeArray<bool>({4, 1}, {true, true, false, false}, true);
    Array b = MakeArray<bool>({4, 1}, {true, false, true, false}, true);
    Array o = a + b;

    auto op_node = o.node()->next_node();
    Array go = MakeArray<bool>({4, 1}, {true, true, true, true});
    Array ga = op_node->backward_functions()[0](go);
    Array gb = op_node->backward_functions()[1](go);

    ExpectEqual<bool>(ga, go);
    ExpectEqual<bool>(gb, go);
}

TEST_P(ArrayTest, MulBackward) {
    Array a = MakeArray<bool>({4, 1}, {true, true, false, false}, true);
    Array b = MakeArray<bool>({4, 1}, {true, false, true, false}, true);
    Array o = a * b;

    auto op_node = o.node()->next_node();
    Array go = MakeArray<bool>({4, 1}, {true, true, true, true});
    Array ga = op_node->backward_functions()[0](go);
    Array gb = op_node->backward_functions()[1](go);

    ExpectEqual<bool>(ga, go * b);
    ExpectEqual<bool>(gb, go * a);
}

TEST_P(ArrayTest, MulBackwrdCapture) {
    Array y = [this]() {
        Array x1 = MakeArray<float>({1}, {2.0f}, true);
        Array x2 = MakeArray<float>({1}, {3.0f}, true);
        return x1 * x2;
    }();
    auto op_node = y.node()->next_node();
    auto lhs_func = op_node->backward_functions()[0];
    auto rhs_func = op_node->backward_functions()[1];
    Array gy = MakeArray<float>({1}, {1.0f});

    Array gx1 = lhs_func(gy);
    Array e1 = MakeArray<float>({1}, {3.0f});
    ExpectEqual<bool>(e1, gx1);

    Array gx2 = rhs_func(gy);
    Array e2 = MakeArray<float>({1}, {2.0f});
    ExpectEqual<bool>(e2, gx2);
}

TEST_P(ArrayTest, MultipleGraphs) {
    Array a = MakeArray<float>({1}, {2.0f}, true, "graph_1");
    Array b = MakeArray<float>({1}, {2.0f}, true, "graph_2");

    EXPECT_TRUE(a.requires_grad("graph_1"));
    EXPECT_TRUE(b.requires_grad("graph_2"));

    Array o = a * b;

    EXPECT_TRUE(o.requires_grad("graph_1"));
    EXPECT_TRUE(o.requires_grad("graph_2"));
    EXPECT_THROW(o.requires_grad("graph_3"), XchainerError);
    EXPECT_THROW(o.requires_grad(""), XchainerError);  // TODO(hvy): Need to fix
}

INSTANTIATE_TEST_CASE_P(ForEachDevice, ArrayTest, ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                                                      std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                                                      std::string{"cpu"}));

}  // namespace
}  // namespace xchainer
