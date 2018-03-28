#include "xchainer/routines/math.h"

#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/array_index.h"
#include "xchainer/check_backward.h"
#include "xchainer/device_id.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/scalar.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/array_check.h"
#include "xchainer/testing/context_session.h"
#include "xchainer/testing/device_session.h"

namespace xchainer {
namespace {

class MathTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        const std::string& backend_name = GetParam();
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

private:
    nonstd::optional<testing::DeviceSession> device_session_;
};

TEST_P(MathTest, IAdd) {
    {
        Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array b = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array e = testing::BuildArray<float>({3, 1}, {2, 4, 6});
        internal::IAdd(a, b);
        testing::ExpectEqual<float>(e, a);
    }

    // non-contiguous
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array a_view = a.At({Slice{}, Slice{1, 2}});
        Array b = Array::OnesLike(a_view);
        Array e_view = testing::BuildArray<int32_t>({3, 1}, {2, 5, 8});
        Array e = testing::BuildArray<int32_t>({3, 3}, {0, 2, 2, 3, 5, 5, 6, 8, 8});
        internal::IAdd(a_view, b);
        testing::ExpectEqual<int32_t>(e_view, a_view);
        testing::ExpectEqual<int32_t>(e, a);
    }

    // broadcast
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3, 1}, Dtype::kInt32);
        Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(1);
        internal::IAdd(a, b);
        testing::ExpectEqual<int32_t>(e, a);
    }
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3}, Dtype::kInt32);
        Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(1);
        internal::IAdd(a, b);
        testing::ExpectEqual<int32_t>(e, a);
    }
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({4}, Dtype::kInt32);
        EXPECT_THROW(internal::IAdd(a, b), XchainerError);
    }
    {
        Array a = testing::BuildArray({3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3, 3}, Dtype::kInt32);
        EXPECT_THROW(internal::IAdd(a, b), XchainerError);
    }
}

TEST_P(MathTest, IMul) {
    {
        Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array b = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array e = testing::BuildArray<float>({3, 1}, {1, 4, 9});
        internal::IMultiply(a, b);
        testing::ExpectEqual<float>(e, a);
    }

    // non-contiguous
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array a_view = a.At({Slice{}, Slice{1, 2}});
        Array b = Array::FullLike(a_view, 2);
        Array e = testing::BuildArray<int32_t>({3, 3}, {0, 2, 2, 3, 8, 5, 6, 14, 8});
        Array e_view = testing::BuildArray<int32_t>({3, 1}, {2, 8, 14});
        internal::IMultiply(a_view, b);
        testing::ExpectEqual<int32_t>(e_view, a_view);
        testing::ExpectEqual<int32_t>(e, a);
    }

    // broadcast
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Full({3, 1}, 2, Dtype::kInt32);
        Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(0, 2);
        internal::IMultiply(a, b);
        testing::ExpectEqual<int32_t>(e, a);
    }
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Full({3}, 2, Dtype::kInt32);
        Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(0, 2);
        internal::IMultiply(a, b);
        testing::ExpectEqual<int32_t>(e, a);
    }
    {
        Array a = testing::BuildArray({3}).WithLinearData<int32_t>();
        Array b = Array::Full({3, 3}, 2, Dtype::kInt32);
        Array e = testing::BuildArray<int32_t>({3, 3}, {0, 2, 4, 0, 2, 4, 0, 2, 4});
        EXPECT_THROW(internal::IMultiply(a, b), XchainerError);
    }
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Full({4}, 2, Dtype::kInt32);
        EXPECT_THROW(internal::IMultiply(a, b), XchainerError);
    }
}

TEST_P(MathTest, Add) {
    {
        Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array b = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array e = testing::BuildArray<float>({3, 1}, {2, 4, 6});
        Array o = Add(a, b);
        testing::ExpectEqual<float>(e, o);
    }

    // non-contiguous
    {
        Array a = Array(testing::BuildArray({3, 3}).WithLinearData<int32_t>()).At({Slice{}, Slice{1, 2}});
        Array b = Array::OnesLike(a);
        Array e = testing::BuildArray<int32_t>({3, 1}, {2, 5, 8});
        Array o = Add(a, b);
        testing::ExpectEqual<int32_t>(e, o);
    }

    // broadcast
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3, 1}, Dtype::kInt32);
        Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(1);
        Array o = Add(a, b);
        testing::ExpectEqual<int32_t>(e, o);
    }
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3}, Dtype::kInt32);
        Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(1);
        Array o = Add(a, b);
        testing::ExpectEqual<int32_t>(e, o);
    }
    {
        Array a = testing::BuildArray({3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3, 3}, Dtype::kInt32);
        Array e = testing::BuildArray<int32_t>({3, 3}, {1, 2, 3, 1, 2, 3, 1, 2, 3});
        Array o = Add(a, b);
        testing::ExpectEqual<int32_t>(e, o);
    }
    {
        Array a = testing::BuildArray({3, 1}).WithLinearData<int32_t>();
        Array b = testing::BuildArray({1, 2}).WithLinearData<int32_t>(1);
        Array e = testing::BuildArray<int32_t>({3, 2}, {1, 2, 2, 3, 3, 4});
        Array o = Add(a, b);
        testing::ExpectEqual<int32_t>(e, o);
    }
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({4}, Dtype::kInt32);
        EXPECT_THROW(Add(a, b), XchainerError);
    }
}

// TODO(hvy): Also test CUDA using MathTest.
TEST(MathNativeTest, MulScalar) {
    testing::ContextSession context_session;

    Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    Array e = testing::BuildArray<float>({3, 1}, {2, 4, 6});
    Array o = Multiply(a, Scalar{2.f});
    testing::ExpectEqual<float>(e, o);
}

TEST_P(MathTest, Mul) {
    {
        Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array b = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array e = testing::BuildArray<float>({3, 1}, {1, 4, 9});
        Array o = Multiply(a, b);
        testing::ExpectEqual<float>(e, o);
    }

    // non-contiguous
    {
        Array a = Array(testing::BuildArray({3, 3}).WithLinearData<int32_t>()).At({Slice{}, Slice{1, 2}});
        Array b = Array::FullLike(a, 2);
        Array e = testing::BuildArray<int32_t>({3, 1}, {2, 8, 14});
        Array o = Multiply(a, b);
        testing::ExpectEqual<int32_t>(e, o);
    }

    // broadcast
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Full({3, 1}, 2, Dtype::kInt32);
        Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(0, 2);
        Array o = Multiply(a, b);
        testing::ExpectEqual<int32_t>(e, o);
    }
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Full({3}, 2, Dtype::kInt32);
        Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(0, 2);
        Array o = Multiply(a, b);
        testing::ExpectEqual<int32_t>(e, o);
    }
    {
        Array a = testing::BuildArray({3}).WithLinearData<int32_t>();
        Array b = Array::Full({3, 3}, 2, Dtype::kInt32);
        Array e = testing::BuildArray<int32_t>({3, 3}, {0, 2, 4, 0, 2, 4, 0, 2, 4});
        Array o = Multiply(a, b);
        testing::ExpectEqual<int32_t>(e, o);
    }
    {
        Array a = testing::BuildArray({3, 1}).WithLinearData<int32_t>(1);
        Array b = testing::BuildArray({1, 2}).WithLinearData<int32_t>(1);
        Array e = testing::BuildArray<int32_t>({3, 2}, {1, 2, 2, 4, 3, 6});
        Array o = Multiply(a, b);
        testing::ExpectEqual<int32_t>(e, o);
    }
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Full({4}, 2, Dtype::kInt32);
        EXPECT_THROW(Multiply(a, b), XchainerError);
    }
}

TEST_P(MathTest, ChainedMath) {
    Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    Array b = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    Array e = testing::BuildArray<float>({3, 1}, {2, 6, 12});
    Array c = Multiply(a, b);
    Array o = Add(a, c);
    testing::ExpectEqual<float>(e, o);
}

TEST_P(MathTest, ChainedInplaceMath) {
    Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    Array b = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    Array e = testing::BuildArray<float>({3, 1}, {2, 6, 12});
    internal::IMultiply(b, a);
    internal::IAdd(a, b);
    testing::ExpectEqual<float>(e, a);
}

TEST_P(MathTest, Sum) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1);
    Array b = Sum(a, std::vector<int8_t>{2, 1, -1});
    EXPECT_EQ(Shape{2}, b.shape());
    Array e = testing::BuildArray(Shape{2}).WithData<T>({630.0f, 1926.0f});
    testing::ExpectEqual<T>(e, b);
}

TEST_P(MathTest, SumAllAxes) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 3}).WithLinearData<T>().WithPadding(1);
    Array b = Sum(a);
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray(Shape{}).WithData<T>({153.0f});
    testing::ExpectEqual<T>(e, b);
}

TEST_P(MathTest, SumZero) {
    using T = float;

    Array a = testing::BuildArray({0}).WithData<T>({});
    Array b = Sum(a);
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray(Shape{}).WithData<T>({0.0f});
    testing::ExpectEqual<T>(e, b);
}

TEST_P(MathTest, SumOne) {
    using T = float;

    Array a = testing::BuildArray({}).WithData<T>({42.0f}).WithPadding(1);
    Array b = Sum(a);
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray(Shape{}).WithData<T>({42.0f});
    testing::ExpectEqual<T>(e, b);
}

TEST_P(MathTest, SumTwo) {
    using T = float;

    Array a = testing::BuildArray({2}).WithData<T>({42.0f, 37.0f}).WithPadding(1);
    Array b = Sum(a);
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray(Shape{}).WithData<T>({79.0f});
    testing::ExpectEqual<T>(e, b);
}

TEST_P(MathTest, SumLarge) {
    using T = int64_t;

    Array a = testing::BuildArray({0x100000}).WithLinearData<T>().WithPadding(1);
    Array b = Sum(a, std::vector<int8_t>{0});
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray(Shape{}).WithData<T>({0x7ffff80000});
    testing::ExpectEqual<T>(e, b);
}

TEST_P(MathTest, SumKeepDims) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 2, 4}).WithLinearData<T>().WithPadding(1);
    Array b = Sum(a, std::vector<int8_t>{-1, 1}, true);
    EXPECT_EQ(Shape({2, 1, 2, 1}), b.shape());
    EXPECT_EQ(0, b.strides()[1]);
    EXPECT_EQ(0, b.strides()[3]);
    Array e = testing::BuildArray(Shape{2, 1, 2, 1}).WithData<T>({114.0f, 162.0f, 402.0f, 450.0f});
    testing::ExpectEqual<T>(e, b);
}

TEST_P(MathTest, InvalidSumDuplicateAxes) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<T>();
    EXPECT_THROW(Sum(a, std::vector<int8_t>{1, 1}), XchainerError);
}

TEST_P(MathTest, InvalidSumOutOfRangeAxes) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<T>();
    EXPECT_THROW(Sum(a, std::vector<int8_t>{3}), DimensionError);
}

TEST_P(MathTest, SumBackward) {
    using T = double;

    CheckBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                return {Sum(xs[0], std::vector<int8_t>{1, 3})};
            },
            {(*testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {testing::BuildArray({2, 4}).WithLinearData<T>(-0.1, 0.1)},
            {Array::Full({2, 3, 4, 3}, 1e-1)});
}

TEST_P(MathTest, SumDoubleBackward_Keepdims) {
    using T = double;

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Sum(xs[0], std::vector<int8_t>{1, 3}, true);
                return {y * y};  // to make it nonlinear
            },
            {(*testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {(*testing::BuildArray({2, 1, 4, 1}).WithLinearData<T>(-0.1, 0.1)).RequireGrad()},
            {testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>()},
            {Array::Full({2, 3, 4, 3}, 1e-1), Array::Full({2, 1, 4, 1}, 1e-1)});
}

TEST_P(MathTest, SumDoubleBackward_NoKeepdims) {
    using T = double;

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Sum(xs[0], std::vector<int8_t>{1, 3}, false);
                return {y * y};  // to make it nonlinear
            },
            {(*testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {(*testing::BuildArray({2, 4}).WithLinearData<T>(-0.1, 0.1)).RequireGrad()},
            {testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>()},
            {Array::Full({2, 3, 4, 3}, 1e-1), Array::Full({2, 4}, 1e-1)});
}

// TODO(hvy): Also test CUDA using MathTest.
TEST(MathTestNative, Maximum) {
    testing::ContextSession context_session;

    Array a = testing::BuildArray<float>({3, 1}, {-1.f, 2.f, -.2f});
    Array e = testing::BuildArray<float>({3, 1}, {0.f, 2.f, 0.f});
    Array b = Maximum(a, Scalar{0.f});
    testing::ExpectEqual<float>(e, b);
}

// TODO(hvy): Also test CUDA using MathTest.
TEST(MathTestNative, MaximumEmpty) {
    testing::ContextSession context_session;

    Array a = testing::BuildArray<float>({0}, {});
    Array e = testing::BuildArray<float>({0}, {});
    Array b = Maximum(a, Scalar{0.f});
    testing::ExpectEqual<float>(e, b);
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        MathTest,
        ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace xchainer
