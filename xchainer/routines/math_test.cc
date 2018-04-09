#include "xchainer/routines/math.h"

#include <cmath>
#include <cstdint>
#include <limits>
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

TEST_P(MathTest, Negative) {
    Array a = testing::BuildArray({3}).WithData<float>({-1, 0, 2});
    Array e = testing::BuildArray({3}).WithData<float>({1, 0, -2});
    Array b = Negative(a);
    testing::ExpectEqual(e, b);
}

TEST_P(MathTest, InvalidNegative) {
    Array a = testing::BuildArray({3}).WithData<bool>({true, false, false});
    EXPECT_THROW(Negative(a), DtypeError);
}

TEST_P(MathTest, NegativeBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(-3).WithPadding(1)).RequireGrad();
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Array::Full(shape, 1e-3);

    CheckBackwardComputation([](const std::vector<Array>& xs) -> std::vector<Array> { return {Negative(xs[0])}; }, {a}, {go}, {eps});
}

TEST_P(MathTest, NegativeDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(-3).WithPadding(1)).RequireGrad();
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Array::Full(shape, 1e-3);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Negative(xs[0]);
                return {y * y};  // to make it nonlinear
            },
            {a},
            {go},
            {ggi},
            {eps, eps});
}

// TODO(niboshi): separate independent tests
TEST_P(MathTest, IAdd) {
    {
        Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array b = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array e = testing::BuildArray<float>({3, 1}, {2, 4, 6});
        internal::IAdd(a, b);
        testing::ExpectEqual(e, a);
    }

    // non-contiguous
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array a_view = a.At({Slice{}, Slice{1, 2}});
        Array b = Array::OnesLike(a_view);
        Array e_view = testing::BuildArray<int32_t>({3, 1}, {2, 5, 8});
        Array e = testing::BuildArray<int32_t>({3, 3}, {0, 2, 2, 3, 5, 5, 6, 8, 8});
        internal::IAdd(a_view, b);
        testing::ExpectEqual(e_view, a_view);
        testing::ExpectEqual(e, a);
    }

    // broadcast
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3, 1}, Dtype::kInt32);
        Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(1);
        internal::IAdd(a, b);
        testing::ExpectEqual(e, a);
    }
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3}, Dtype::kInt32);
        Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(1);
        internal::IAdd(a, b);
        testing::ExpectEqual(e, a);
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

// TODO(niboshi): separate independent tests
TEST_P(MathTest, ISubtract) {
    {
        Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array b = testing::BuildArray<float>({3, 1}, {4, 0, -2});
        Array e = testing::BuildArray<float>({3, 1}, {-3, 2, 5});
        internal::ISubtract(a, b);
        testing::ExpectEqual(e, a);
    }

    // non-contiguous
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array a_view = a.At({Slice{}, Slice{1, 2}});
        Array b = Array::OnesLike(a_view);
        Array e_view = testing::BuildArray<int32_t>({3, 1}, {0, 3, 6});
        Array e = testing::BuildArray<int32_t>({3, 3}, {0, 0, 2, 3, 3, 5, 6, 6, 8});
        internal::ISubtract(a_view, b);
        testing::ExpectEqual(e_view, a_view);
        testing::ExpectEqual(e, a);
    }

    // broadcast
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3, 1}, Dtype::kInt32);
        Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(-1);
        internal::ISubtract(a, b);
        testing::ExpectEqual(e, a);
    }
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3}, Dtype::kInt32);
        Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(-1);
        internal::ISubtract(a, b);
        testing::ExpectEqual(e, a);
    }
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({4}, Dtype::kInt32);
        EXPECT_THROW(internal::ISubtract(a, b), XchainerError);
    }
    {
        Array a = testing::BuildArray({3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3, 3}, Dtype::kInt32);
        EXPECT_THROW(internal::ISubtract(a, b), XchainerError);
    }
}

// TODO(niboshi): separate independent tests
TEST_P(MathTest, IMultiply) {
    {
        Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array b = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array e = testing::BuildArray<float>({3, 1}, {1, 4, 9});
        internal::IMultiply(a, b);
        testing::ExpectEqual(e, a);
    }

    // non-contiguous
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array a_view = a.At({Slice{}, Slice{1, 2}});
        Array b = Array::FullLike(a_view, 2);
        Array e = testing::BuildArray<int32_t>({3, 3}, {0, 2, 2, 3, 8, 5, 6, 14, 8});
        Array e_view = testing::BuildArray<int32_t>({3, 1}, {2, 8, 14});
        internal::IMultiply(a_view, b);
        testing::ExpectEqual(e_view, a_view);
        testing::ExpectEqual(e, a);
    }

    // broadcast
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Full({3, 1}, 2, Dtype::kInt32);
        Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(0, 2);
        internal::IMultiply(a, b);
        testing::ExpectEqual(e, a);
    }
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Full({3}, 2, Dtype::kInt32);
        Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(0, 2);
        internal::IMultiply(a, b);
        testing::ExpectEqual(e, a);
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

TEST_P(MathTest, IDivide) {
    Array a = testing::BuildArray<float>({3, 1}, {-3, -3, 0}).WithPadding(1);
    Array b = testing::BuildArray<float>({3, 1}, {2, -2, 1}).WithPadding(2);
    Array e = testing::BuildArray<float>({3, 1}, {-1.5f, 1.5f, 0});
    internal::IDivide(a, b);
    testing::ExpectEqual(e, a);
}

TEST_P(MathTest, IDivideBroadcast1) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<float>().WithPadding(1);
    Array b = Array::Full({3, 1}, 2.0f, Dtype::kFloat32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<float>(0, 0.5f);
    internal::IDivide(a, b);
    testing::ExpectEqual(e, a);
}

TEST_P(MathTest, IDivideBroacast2) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<float>().WithPadding(1);
    Array b = Array::Full({3}, 2.0f, Dtype::kFloat32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<float>(0, 0.5f);
    internal::IDivide(a, b);
    testing::ExpectEqual(e, a);
}

TEST_P(MathTest, IDivideInvalidBroadcast1) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<float>();
    Array b = Array::Full({4}, 2.0f, Dtype::kFloat32);
    EXPECT_THROW(internal::IDivide(a, b), XchainerError);
}

TEST_P(MathTest, IDivideInvalidBroadcast2) {
    Array a = testing::BuildArray({3}).WithLinearData<float>();
    Array b = Array::Ones({3, 3}, Dtype::kFloat32);
    EXPECT_THROW(internal::IDivide(a, b), XchainerError);
}

// TODO(niboshi): Write backward and double-backward tests for add/subtract/mul

// TODO(niboshi): separate independent tests
TEST_P(MathTest, Add) {
    {
        Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array b = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array e = testing::BuildArray<float>({3, 1}, {2, 4, 6});
        Array o = Add(a, b);
        testing::ExpectEqual(e, o);
    }

    // non-contiguous
    {
        Array a = Array(testing::BuildArray({3, 3}).WithLinearData<int32_t>()).At({Slice{}, Slice{1, 2}});
        Array b = Array::OnesLike(a);
        Array e = testing::BuildArray<int32_t>({3, 1}, {2, 5, 8});
        Array o = Add(a, b);
        testing::ExpectEqual(e, o);
    }

    // broadcast
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3, 1}, Dtype::kInt32);
        Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(1);
        Array o = Add(a, b);
        testing::ExpectEqual(e, o);
    }
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3}, Dtype::kInt32);
        Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(1);
        Array o = Add(a, b);
        testing::ExpectEqual(e, o);
    }
    {
        Array a = testing::BuildArray({3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3, 3}, Dtype::kInt32);
        Array e = testing::BuildArray<int32_t>({3, 3}, {1, 2, 3, 1, 2, 3, 1, 2, 3});
        Array o = Add(a, b);
        testing::ExpectEqual(e, o);
    }
    {
        Array a = testing::BuildArray({3, 1}).WithLinearData<int32_t>();
        Array b = testing::BuildArray({1, 2}).WithLinearData<int32_t>(1);
        Array e = testing::BuildArray<int32_t>({3, 2}, {1, 2, 2, 3, 3, 4});
        Array o = Add(a, b);
        testing::ExpectEqual(e, o);
    }
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({4}, Dtype::kInt32);
        EXPECT_THROW(Add(a, b), XchainerError);
    }
}

TEST_P(MathTest, Subtract) {
    Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3}).WithPadding(1);
    Array b = testing::BuildArray<float>({3, 1}, {4, 0, -2}).WithPadding(2);
    Array e = testing::BuildArray<float>({3, 1}, {-3, 2, 5});
    Array o = Subtract(a, b);
    testing::ExpectEqual(e, o);
}

TEST_P(MathTest, SubtractBroadcast1) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Array::Ones({3, 1}, Dtype::kInt32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(-1);
    Array o = Subtract(a, b);
    testing::ExpectEqual(e, o);
}

TEST_P(MathTest, SubtractBroadcast2) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Array::Ones({3}, Dtype::kInt32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(-1);
    Array o = Subtract(a, b);
    testing::ExpectEqual(e, o);
}

TEST_P(MathTest, SubtractBroadcast3) {
    Array a = testing::BuildArray({3}).WithLinearData<int32_t>();
    Array b = Array::Ones({3, 3}, Dtype::kInt32);
    Array e = testing::BuildArray<int32_t>({3, 3}, {-1, 0, 1, -1, 0, 1, -1, 0, 1});
    Array o = Subtract(a, b);
    testing::ExpectEqual(e, o);
}

TEST_P(MathTest, SubtractBroadcast4) {
    Array a = testing::BuildArray({3, 1}).WithLinearData<int32_t>();
    Array b = testing::BuildArray({1, 2}).WithLinearData<int32_t>(1);
    Array e = testing::BuildArray<int32_t>({3, 2}, {-1, -2, 0, -1, 1, 0});
    Array o = Subtract(a, b);
    testing::ExpectEqual(e, o);
}

TEST_P(MathTest, SubtractInvalidBroadcast) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Array::Ones({4}, Dtype::kInt32);
    EXPECT_THROW(Subtract(a, b), XchainerError);
}

TEST_P(MathTest, MultiplyScalar) {
    Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    Array e = testing::BuildArray<float>({3, 1}, {2, 4, 6});

    // array * scalar
    {
        Array o = Multiply(a, Scalar{2.f});
        testing::ExpectEqual(e, o);
    }
    // scalar * array
    {
        Array o = Multiply(Scalar{2.f}, a);
        testing::ExpectEqual(e, o);
    }
}

TEST_P(MathTest, MultiplyScalarBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Scalar s{T{2.0}};
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Array::Full(shape, 1e-1);

    // array * scalar
    CheckBackwardComputation([s](const std::vector<Array>& xs) -> std::vector<Array> { return {Multiply(xs[0], s)}; }, {a}, {go}, {eps});
    // scalar * array
    CheckBackwardComputation([s](const std::vector<Array>& xs) -> std::vector<Array> { return {Multiply(s, xs[0])}; }, {a}, {go}, {eps});
}

TEST_P(MathTest, MultiplyScalarDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Scalar s{T{2.0}};
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.3, 0.1).WithPadding(1);
    Array eps = Array::Full(shape, 1e-1);

    // array * scalar
    CheckDoubleBackwardComputation(
            [s](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Multiply(xs[0], s);
                return {y * y};  // to make it nonlinear
            },
            {a},
            {go},
            {ggi},
            {eps, eps});
    // scalar * array
    CheckDoubleBackwardComputation(
            [s](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Multiply(s, xs[0]);
                return {y * y};  // to make it nonlinear
            },
            {a},
            {go},
            {ggi},
            {eps, eps});
}

// TODO(niboshi): separate independent tests
TEST_P(MathTest, Multiply) {
    {
        Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array b = testing::BuildArray<float>({3, 1}, {1, 2, 3});
        Array e = testing::BuildArray<float>({3, 1}, {1, 4, 9});
        Array o = Multiply(a, b);
        testing::ExpectEqual(e, o);
    }

    // non-contiguous
    {
        Array a = Array(testing::BuildArray({3, 3}).WithLinearData<int32_t>()).At({Slice{}, Slice{1, 2}});
        Array b = Array::FullLike(a, 2);
        Array e = testing::BuildArray<int32_t>({3, 1}, {2, 8, 14});
        Array o = Multiply(a, b);
        testing::ExpectEqual(e, o);
    }

    // broadcast
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Full({3, 1}, 2, Dtype::kInt32);
        Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(0, 2);
        Array o = Multiply(a, b);
        testing::ExpectEqual(e, o);
    }
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Full({3}, 2, Dtype::kInt32);
        Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(0, 2);
        Array o = Multiply(a, b);
        testing::ExpectEqual(e, o);
    }
    {
        Array a = testing::BuildArray({3}).WithLinearData<int32_t>();
        Array b = Array::Full({3, 3}, 2, Dtype::kInt32);
        Array e = testing::BuildArray<int32_t>({3, 3}, {0, 2, 4, 0, 2, 4, 0, 2, 4});
        Array o = Multiply(a, b);
        testing::ExpectEqual(e, o);
    }
    {
        Array a = testing::BuildArray({3, 1}).WithLinearData<int32_t>(1);
        Array b = testing::BuildArray({1, 2}).WithLinearData<int32_t>(1);
        Array e = testing::BuildArray<int32_t>({3, 2}, {1, 2, 2, 4, 3, 6});
        Array o = Multiply(a, b);
        testing::ExpectEqual(e, o);
    }
    {
        Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Full({4}, 2, Dtype::kInt32);
        EXPECT_THROW(Multiply(a, b), XchainerError);
    }
}

TEST_P(MathTest, Divide) {
    Array a = testing::BuildArray<float>({3, 1}, {-3, -3, 0}).WithPadding(1);
    Array b = testing::BuildArray<float>({3, 1}, {2, -2, 1}).WithPadding(2);
    Array e = testing::BuildArray<float>({3, 1}, {-1.5f, 1.5f, 0});
    Array o = Divide(a, b);
    testing::ExpectEqual(e, o);
}

TEST_P(MathTest, DivideBroadcast1) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<float>().WithPadding(1);
    Array b = Array::Full({3, 1}, 2.0f, Dtype::kFloat32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<float>(0, 0.5f);
    Array o = Divide(a, b);
    testing::ExpectEqual(e, o);
}

TEST_P(MathTest, DivideBroadcast2) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<float>().WithPadding(1);
    Array b = Array::Full({3}, 2.0f, Dtype::kFloat32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<float>(0, 0.5f);
    Array o = Divide(a, b);
    testing::ExpectEqual(e, o);
}

TEST_P(MathTest, DivideBroadcast3) {
    Array a = testing::BuildArray({3}).WithLinearData<float>().WithPadding(1);
    Array b = Array::Full({3, 3}, 2.0f, Dtype::kFloat32);
    Array e = testing::BuildArray({3, 3}).WithData<float>({0.0f, 0.5f, 1.0f, 0.0f, 0.5f, 1.0f, 0.0f, 0.5f, 1.0f});
    Array o = Divide(a, b);
    testing::ExpectEqual(e, o);
}

TEST_P(MathTest, DivideBroadcast4) {
    Array a = testing::BuildArray({3, 1}).WithLinearData<float>().WithPadding(1);
    Array b = Array::Full({1, 2}, 2.0f, Dtype::kFloat32);
    Array e = testing::BuildArray({3, 2}).WithData<float>({0.0f, 0.0f, 0.5f, 0.5f, 1.0f, 1.0f});
    Array o = Divide(a, b);
    testing::ExpectEqual(e, o);
}

TEST_P(MathTest, DivideInvalidBroadcast) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<float>();
    Array b = Array::Ones({4}, Dtype::kFloat32);
    EXPECT_THROW(Divide(a, b), XchainerError);
}

TEST_P(MathTest, DivideBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(-2).WithPadding(1)).RequireGrad();
    Array b = (*testing::BuildArray(shape).WithData<T>({-6, -4, -2, 2, 4, 6}).WithPadding(2)).RequireGrad();
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(3);
    Array eps = Array::Full(shape, 1e-3);

    CheckBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {Divide(xs[0], xs[1])}; }, {a, b}, {go}, {eps, eps});
}

TEST_P(MathTest, DivideDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(-2).WithPadding(1)).RequireGrad();
    Array b = (*testing::BuildArray(shape).WithData<T>({-6, -4, -2, 2, 4, 6}).WithPadding(2)).RequireGrad();
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(3)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.3, 0.1).WithPadding(4);
    Array eps = Array::Full(shape, 1e-3);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {Divide(xs[0], xs[1])}; },
            {a, b},
            {go},
            {ggi, ggi},
            {eps, eps, eps});
}

TEST_P(MathTest, ChainedMath) {
    Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    Array b = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    Array e = testing::BuildArray<float>({3, 1}, {2, 6, 12});
    Array c = Multiply(a, b);
    Array o = Add(a, c);
    testing::ExpectEqual(e, o);
}

TEST_P(MathTest, ChainedInplaceMath) {
    Array a = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    Array b = testing::BuildArray<float>({3, 1}, {1, 2, 3});
    Array e = testing::BuildArray<float>({3, 1}, {2, 6, 12});
    internal::IMultiply(b, a);
    internal::IAdd(a, b);
    testing::ExpectEqual(e, a);
}

TEST_P(MathTest, Sum) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1);
    Array b = Sum(a, std::vector<int8_t>{2, 1, -1});
    EXPECT_EQ(Shape{2}, b.shape());
    Array e = testing::BuildArray(Shape{2}).WithData<T>({630.0f, 1926.0f});
    testing::ExpectEqual(e, b);
}

TEST_P(MathTest, SumAllAxes) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 3}).WithLinearData<T>().WithPadding(1);
    Array b = Sum(a);
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray(Shape{}).WithData<T>({153.0f});
    testing::ExpectEqual(e, b);
}

TEST_P(MathTest, SumZero) {
    using T = float;

    Array a = testing::BuildArray({0}).WithData<T>({});
    Array b = Sum(a);
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray(Shape{}).WithData<T>({0.0f});
    testing::ExpectEqual(e, b);
}

TEST_P(MathTest, SumOne) {
    using T = float;

    Array a = testing::BuildArray({}).WithData<T>({42.0f}).WithPadding(1);
    Array b = Sum(a);
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray(Shape{}).WithData<T>({42.0f});
    testing::ExpectEqual(e, b);
}

TEST_P(MathTest, SumTwo) {
    using T = float;

    Array a = testing::BuildArray({2}).WithData<T>({42.0f, 37.0f}).WithPadding(1);
    Array b = Sum(a);
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray(Shape{}).WithData<T>({79.0f});
    testing::ExpectEqual(e, b);
}

TEST_P(MathTest, SumLarge) {
    using T = int64_t;

    Array a = testing::BuildArray({0x100000}).WithLinearData<T>().WithPadding(1);
    Array b = Sum(a, std::vector<int8_t>{0});
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray(Shape{}).WithData<T>({0x7ffff80000});
    testing::ExpectEqual(e, b);
}

TEST_P(MathTest, SumKeepDims) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 2, 4}).WithLinearData<T>().WithPadding(1);
    Array b = Sum(a, std::vector<int8_t>{-1, 1}, true);
    EXPECT_EQ(Shape({2, 1, 2, 1}), b.shape());
    EXPECT_EQ(0, b.strides()[1]);
    EXPECT_EQ(0, b.strides()[3]);
    Array e = testing::BuildArray(Shape{2, 1, 2, 1}).WithData<T>({114.0f, 162.0f, 402.0f, 450.0f});
    testing::ExpectEqual(e, b);
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

TEST_P(MathTest, MaximumScalar) {
    Array a = testing::BuildArray<float>({3, 1}, {-1.f, 2.f, -.2f});
    Array e = testing::BuildArray<float>({3, 1}, {0.f, 2.f, 0.f});

    {
        Array b = Maximum(a, Scalar{0.f});
        testing::ExpectEqual(e, b);
    }
    {
        Array b = Maximum(Scalar{0.f}, a);
        testing::ExpectEqual(e, b);
    }
}

TEST_P(MathTest, MaximumScalarEmpty) {
    Array a = testing::BuildArray<float>({0}, {});
    Array e = testing::BuildArray<float>({0}, {});
    Array b = Maximum(a, Scalar{0.f});
    testing::ExpectEqual(e, b);
}

TEST_P(MathTest, MaximumScalarBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Scalar s{T{0.2}};
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Array::Full(shape, 1e-1);

    // Maximum(array, scalar)
    CheckBackwardComputation([s](const std::vector<Array>& xs) -> std::vector<Array> { return {Maximum(xs[0], s)}; }, {a}, {go}, {eps});
    // Maximum(scalar, array)
    CheckBackwardComputation([s](const std::vector<Array>& xs) -> std::vector<Array> { return {Maximum(s, xs[0])}; }, {a}, {go}, {eps});
}

TEST_P(MathTest, MaximumScalarDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Scalar s{T{0.2}};
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.3, 0.1).WithPadding(1);
    Array eps = Array::Full(shape, 1e-1);

    // Maximum(array, scalar)
    CheckDoubleBackwardComputation(
            [s](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Maximum(xs[0], s);
                return {y * y};  // to make it nonlinear
            },
            {a},
            {go},
            {ggi},
            {eps, eps});
    // Maximum(scalar, array)
    CheckDoubleBackwardComputation(
            [s](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Maximum(s, xs[0]);
                return {y * y};  // to make it nonlinear
            },
            {a},
            {go},
            {ggi},
            {eps, eps});
}

TEST_P(MathTest, Exp) {
    Array a = testing::BuildArray<float>(
            {5}, {0.f, 1.f, std::log(3.f), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()});
    Array e = testing::BuildArray<float>({5}, {1.f, std::exp(1.f), 3.f, std::numeric_limits<float>::infinity(), 0});
    Array b = Exp(a);
    testing::ExpectAllClose(e, b, 1e-3, 0);
}

TEST_P(MathTest, ExpBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Array::Full(shape, 1e-3);

    CheckBackwardComputation([](const std::vector<Array>& xs) -> std::vector<Array> { return {Exp(xs[0])}; }, {a}, {go}, {eps});
}

TEST_P(MathTest, ExpDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Array::Full(shape, 1e-3);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {Exp(xs[0])}; }, {a}, {go}, {ggi}, {eps, eps});
}

TEST_P(MathTest, Log) {
    Array a = testing::BuildArray<float>({6}, {0.0f, 1.0f, 3.0f, -1.f, std::exp(-4.0f), std::exp(4.0f)}).WithPadding(1);
    Array e = testing::BuildArray<float>({6}, {-std::numeric_limits<float>::infinity(), 0.0f, std::log(3.0f), std::nanf(""), -4.0f, 4.0f});
    Array b = Log(a);
    testing::ExpectAllClose(e, b, 1e-3, 0, true);
}

TEST_P(MathTest, LogBackward) {
    // TODO(niboshi): Implement
}

TEST_P(MathTest, LogDoubleBackward) {
    // TODO(niboshi): Implement
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
