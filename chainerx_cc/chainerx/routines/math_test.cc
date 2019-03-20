#include "chainerx/routines/math.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/array_index.h"
#include "chainerx/axes.h"
#include "chainerx/check_backward.h"
#include "chainerx/device_id.h"
#include "chainerx/dtype.h"
#include "chainerx/error.h"
#include "chainerx/routines/creation.h"
#include "chainerx/scalar.h"
#include "chainerx/testing/array.h"
#include "chainerx/testing/array_check.h"
#include "chainerx/testing/device_session.h"
#include "chainerx/testing/routines.h"
#include "chainerx/testing/threading.h"

namespace chainerx {
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

TEST_THREAD_SAFE_P(MathTest, Negative) {
    Array a = testing::BuildArray({3}).WithData<float>({-1, 0, 2});
    Array e = testing::BuildArray({3}).WithData<float>({1, 0, -2});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Negative(xs[0])}; }, {a}, {e}); });
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
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {Negative(xs[0])}; }, {a}, {go}, {eps});
}

TEST_P(MathTest, NegativeDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(-3).WithPadding(1)).RequireGrad();
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

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

TEST_P(MathTest, IAdd) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Array b = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Array e = testing::BuildArray({3, 1}).WithData<float>({2, 4, 6});
    internal::IAdd(a, b);
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, IAddNonContiguous) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array a_view = a.At({Slice{}, Slice{1, 2}});
    Array b = OnesLike(a_view);
    Array e_view = testing::BuildArray({3, 1}).WithData<int32_t>({2, 5, 8});
    Array e = testing::BuildArray({3, 3}).WithData<int32_t>({0, 2, 2, 3, 5, 5, 6, 8, 8});
    internal::IAdd(a_view, b);
    EXPECT_ARRAY_EQ(e_view, a_view);
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, IAddBroadcast1) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Ones({3, 1}, Dtype::kInt32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(1);
    internal::IAdd(a, b);
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, IAddBroadcast2) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Ones({3}, Dtype::kInt32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(1);
    internal::IAdd(a, b);
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, IAddInvalidBroadcast1) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Ones({4}, Dtype::kInt32);
    EXPECT_THROW(internal::IAdd(a, b), ChainerxError);
}

TEST_P(MathTest, IAddInvalidBroadcast2) {
    Array a = testing::BuildArray({3}).WithLinearData<int32_t>();
    Array b = Ones({3, 3}, Dtype::kInt32);
    EXPECT_THROW(internal::IAdd(a, b), ChainerxError);
}

TEST_P(MathTest, IAddScalar) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Array e = testing::BuildArray({3, 1}).WithData<float>({3, 4, 5});
    internal::IAdd(a, Scalar{2.f});
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, ISubtract) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Array b = testing::BuildArray({3, 1}).WithData<float>({4, 0, -2});
    Array e = testing::BuildArray({3, 1}).WithData<float>({-3, 2, 5});
    internal::ISubtract(a, b);
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, ISubtractNonContiguous) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array a_view = a.At({Slice{}, Slice{1, 2}});
    Array b = OnesLike(a_view);
    Array e_view = testing::BuildArray({3, 1}).WithData<int32_t>({0, 3, 6});
    Array e = testing::BuildArray({3, 3}).WithData<int32_t>({0, 0, 2, 3, 3, 5, 6, 6, 8});
    internal::ISubtract(a_view, b);
    EXPECT_ARRAY_EQ(e_view, a_view);
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, ISubtractBroadcast1) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Ones({3, 1}, Dtype::kInt32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(-1);
    internal::ISubtract(a, b);
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, ISubtractBroadcast2) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Ones({3}, Dtype::kInt32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(-1);
    internal::ISubtract(a, b);
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, ISubtractInvalidBroadcast1) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Ones({4}, Dtype::kInt32);
    EXPECT_THROW(internal::ISubtract(a, b), ChainerxError);
}

TEST_P(MathTest, ISubtractInvalidBroadcast2) {
    Array a = testing::BuildArray({3}).WithLinearData<int32_t>();
    Array b = Ones({3, 3}, Dtype::kInt32);
    EXPECT_THROW(internal::ISubtract(a, b), ChainerxError);
}

TEST_P(MathTest, ISubtractInvalidDtype) {
    Array a = Ones({3}, Dtype::kBool);
    Array b = Ones({3}, Dtype::kBool);
    EXPECT_THROW(internal::ISubtract(a, b), DtypeError);
}

TEST_P(MathTest, ISubtractScalar) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Array e = testing::BuildArray({3, 1}).WithData<float>({-2, -1, 0});
    internal::ISubtract(a, Scalar{3.f});
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, ISubtractScalarInvalidDtype) {
    Array a = Ones({3}, Dtype::kBool);
    EXPECT_THROW(internal::ISubtract(a, Scalar{true}), DtypeError);
}

TEST_P(MathTest, IMultiply) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Array b = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Array e = testing::BuildArray({3, 1}).WithData<float>({1, 4, 9});
    internal::IMultiply(a, b);
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, IMultiplyNonContiguous) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array a_view = a.At({Slice{}, Slice{1, 2}});
    Array b = FullLike(a_view, 2);
    Array e = testing::BuildArray({3, 3}).WithData<int32_t>({0, 2, 2, 3, 8, 5, 6, 14, 8});
    Array e_view = testing::BuildArray({3, 1}).WithData<int32_t>({2, 8, 14});
    internal::IMultiply(a_view, b);
    EXPECT_ARRAY_EQ(e_view, a_view);
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, IMultiplyBroadcast1) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Full({3, 1}, 2, Dtype::kInt32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(0, 2);
    internal::IMultiply(a, b);
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, IMultiplyBroadcast2) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Full({3}, 2, Dtype::kInt32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(0, 2);
    internal::IMultiply(a, b);
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, IMultiplyInvalidBroadcast1) {
    Array a = testing::BuildArray({3}).WithLinearData<int32_t>();
    Array b = Full({3, 3}, 2, Dtype::kInt32);
    Array e = testing::BuildArray({3, 3}).WithData<int32_t>({0, 2, 4, 0, 2, 4, 0, 2, 4});
    EXPECT_THROW(internal::IMultiply(a, b), ChainerxError);
}

TEST_P(MathTest, IMultiplyInvalidBroadcast2) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Full({4}, 2, Dtype::kInt32);
    EXPECT_THROW(internal::IMultiply(a, b), ChainerxError);
}

TEST_P(MathTest, IMultiplyScalar) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Array e = testing::BuildArray({3, 1}).WithData<float>({2, 4, 6});
    internal::IMultiply(a, Scalar{2.f});
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, IDivide) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({-3, -3, 0}).WithPadding(1);
    Array b = testing::BuildArray({3, 1}).WithData<float>({2, -2, 1}).WithPadding(2);
    Array e = testing::BuildArray({3, 1}).WithData<float>({-1.5f, 1.5f, 0});
    internal::IDivide(a, b);
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, IDivideNonContiguous) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<float>();
    Array a_view = a.At({Slice{}, Slice{1, 2}});
    Array b = FullLike(a_view, 2);
    Array e = testing::BuildArray({3, 3}).WithData<float>({0.f, 0.5f, 2.f, 3.f, 2.f, 5.f, 6.f, 3.5f, 8.f});
    Array e_view = testing::BuildArray({3, 1}).WithData<float>({0.5f, 2.f, 3.5f});
    internal::IDivide(a_view, b);
    EXPECT_ARRAY_EQ(e_view, a_view);
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, IDivideBroadcast1) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<float>().WithPadding(1);
    Array b = Full({3, 1}, 2.0f, Dtype::kFloat32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<float>(0, 0.5f);
    internal::IDivide(a, b);
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, IDivideBroacast2) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<float>().WithPadding(1);
    Array b = Full({3}, 2.0f, Dtype::kFloat32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<float>(0, 0.5f);
    internal::IDivide(a, b);
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, IDivideInvalidBroadcast1) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<float>();
    Array b = Full({4}, 2.0f, Dtype::kFloat32);
    EXPECT_THROW(internal::IDivide(a, b), ChainerxError);
}

TEST_P(MathTest, IDivideInvalidBroadcast2) {
    Array a = testing::BuildArray({3}).WithLinearData<float>();
    Array b = Ones({3, 3}, Dtype::kFloat32);
    EXPECT_THROW(internal::IDivide(a, b), ChainerxError);
}

TEST_P(MathTest, IDivideScalar) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({1.f, 2.f, 3.f});
    Array e = testing::BuildArray({3, 1}).WithData<float>({0.5f, 1.f, 1.5f});
    internal::IDivide(a, Scalar{2.f});
    EXPECT_ARRAY_EQ(e, a);
}

TEST_P(MathTest, IDivideInteger) {
    Array a = testing::BuildArray({3, 1}).WithData<int64_t>({2, 4, 6});
    Array b = testing::BuildArray({3, 1}).WithData<int64_t>({1, 2, 3});
    EXPECT_THROW(internal::IDivide(a, b), DtypeError);
}

TEST_P(MathTest, IDivideScalarInteger) {
    Array a = testing::BuildArray({3, 1}).WithData<int64_t>({1, 2, 3});
    EXPECT_THROW(internal::IDivide(a, Scalar{1}), DtypeError);
}

TEST_THREAD_SAFE_P(MathTest, Add) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Array b = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Array e = testing::BuildArray({3, 1}).WithData<float>({2, 4, 6});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Add(xs[0], xs[1])}; }, {a, b}, {e}); });
}

TEST_THREAD_SAFE_P(MathTest, AddNonContiguous) {
    Array a = Array(testing::BuildArray({3, 3}).WithLinearData<int32_t>()).At({Slice{}, Slice{1, 2}});
    Array b = OnesLike(a);
    Array e = testing::BuildArray({3, 1}).WithData<int32_t>({2, 5, 8});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Add(xs[0], xs[1])}; }, {a, b}, {e}); });
}

TEST_THREAD_SAFE_P(MathTest, AddBroadcast1) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Ones({3, 1}, Dtype::kInt32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(1);

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Add(xs[0], xs[1])}; }, {a, b}, {e}); });
}

TEST_THREAD_SAFE_P(MathTest, AddBroadcast2) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Ones({3}, Dtype::kInt32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(1);

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Add(xs[0], xs[1])}; }, {a, b}, {e}); });
}

TEST_THREAD_SAFE_P(MathTest, AddBroadcast3) {
    Array a = testing::BuildArray({3}).WithLinearData<int32_t>();
    Array b = Ones({3, 3}, Dtype::kInt32);
    Array e = testing::BuildArray({3, 3}).WithData<int32_t>({1, 2, 3, 1, 2, 3, 1, 2, 3});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Add(xs[0], xs[1])}; }, {a, b}, {e}); });
}

TEST_THREAD_SAFE_P(MathTest, AddBroadcast4) {
    Array a = testing::BuildArray({3, 1}).WithLinearData<int32_t>();
    Array b = testing::BuildArray({1, 2}).WithLinearData<int32_t>(1);
    Array e = testing::BuildArray({3, 2}).WithData<int32_t>({1, 2, 2, 3, 3, 4});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Add(xs[0], xs[1])}; }, {a, b}, {e}); });
}

TEST_P(MathTest, AddInvalidBroadcast) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Ones({4}, Dtype::kInt32);
    EXPECT_THROW(Add(a, b), ChainerxError);
}

TEST_THREAD_SAFE_P(MathTest, AddArrayScalar) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Scalar b{2.f};
    Array e = testing::BuildArray({3, 1}).WithData<float>({3, 4, 5});

    Run([&]() { testing::CheckForward([&b](const std::vector<Array>& xs) { return std::vector<Array>{Add(xs[0], b)}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(MathTest, AddScalarArray) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Scalar b{2.f};
    Array e = testing::BuildArray({3, 1}).WithData<float>({3, 4, 5});

    Run([&]() { testing::CheckForward([&b](const std::vector<Array>& xs) { return std::vector<Array>{Add(b, xs[0])}; }, {a}, {e}); });
}

TEST_P(MathTest, AddBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(-2).WithPadding(1)).RequireGrad();
    Array b = (*testing::BuildArray(shape).WithData<T>({-6, -4, -2, 2, 4, 6}).WithPadding(2)).RequireGrad();
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(3);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {Add(xs[0], xs[1])}; }, {a, b}, {go}, {eps, eps});
}

TEST_P(MathTest, AddDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(-2).WithPadding(1)).RequireGrad();
    Array b = (*testing::BuildArray(shape).WithData<T>({-6, -4, -2, 2, 4, 6}).WithPadding(2)).RequireGrad();
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(3)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.3, 0.1).WithPadding(4);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Add(xs[0], xs[1]);
                return {y * y};  // to make it nonlinear
            },
            {a, b},
            {go},
            {ggi, ggi},
            {eps, eps, eps});
}

TEST_P(MathTest, AddScalarBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Scalar b{T{2.0}};
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-1, Dtype::kFloat64);

    // array + scalar
    CheckBackward([b](const std::vector<Array>& xs) -> std::vector<Array> { return {Add(xs[0], b)}; }, {a}, {go}, {eps});
    // scalar + array
    CheckBackward([b](const std::vector<Array>& xs) -> std::vector<Array> { return {Add(b, xs[0])}; }, {a}, {go}, {eps});
}

TEST_P(MathTest, AddScalarDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Scalar b{T{2.0}};
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.3, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-1, Dtype::kFloat64);

    // array + scalar
    CheckDoubleBackwardComputation(
            [b](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Add(xs[0], b);
                return {y * y};  // to make it nonlinear
            },
            {a},
            {go},
            {ggi},
            {eps, eps});
    // scalar + array
    CheckDoubleBackwardComputation(
            [b](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Add(b, xs[0]);
                return {y * y};  // to make it nonlinear
            },
            {a},
            {go},
            {ggi},
            {eps, eps});
}

TEST_THREAD_SAFE_P(MathTest, Subtract) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3}).WithPadding(1);
    Array b = testing::BuildArray({3, 1}).WithData<float>({4, 0, -2}).WithPadding(2);
    Array e = testing::BuildArray({3, 1}).WithData<float>({-3, 2, 5});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Subtract(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, SubtractNonContiguous) {
    Array a = Array(testing::BuildArray({3, 3}).WithLinearData<int32_t>()).At({Slice{}, Slice{1, 2}});
    Array b = OnesLike(a);
    Array e = testing::BuildArray({3, 1}).WithData<int32_t>({0, 3, 6});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Subtract(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, SubtractBroadcast1) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Ones({3, 1}, Dtype::kInt32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(-1);

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Subtract(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, SubtractBroadcast2) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Ones({3}, Dtype::kInt32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(-1);

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Subtract(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, SubtractBroadcast3) {
    Array a = testing::BuildArray({3}).WithLinearData<int32_t>();
    Array b = Ones({3, 3}, Dtype::kInt32);
    Array e = testing::BuildArray({3, 3}).WithData<int32_t>({-1, 0, 1, -1, 0, 1, -1, 0, 1});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Subtract(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, SubtractBroadcast4) {
    Array a = testing::BuildArray({3, 1}).WithLinearData<int32_t>();
    Array b = testing::BuildArray({1, 2}).WithLinearData<int32_t>(1);
    Array e = testing::BuildArray({3, 2}).WithData<int32_t>({-1, -2, 0, -1, 1, 0});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Subtract(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_P(MathTest, SubtractInvalidBroadcast) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Ones({4}, Dtype::kInt32);
    EXPECT_THROW(Subtract(a, b), ChainerxError);
}

TEST_P(MathTest, SubtractInvalidDtype) {
    Array a = Ones({3}, Dtype::kBool);
    Array b = Ones({3}, Dtype::kBool);
    EXPECT_THROW(Subtract(a, b), DtypeError);
}

TEST_THREAD_SAFE_P(MathTest, SubtractArrayScalar) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Scalar b{2.f};

    Array e = testing::BuildArray({3, 1}).WithData<float>({-1, 0, 1});
    Run([&]() { testing::CheckForward([&b](const std::vector<Array>& xs) { return std::vector<Array>{Subtract(xs[0], b)}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(MathTest, SubtractScalarArray) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Scalar b{2.f};

    Array e = testing::BuildArray({3, 1}).WithData<float>({1, 0, -1});
    Run([&]() { testing::CheckForward([&b](const std::vector<Array>& xs) { return std::vector<Array>{Subtract(b, xs[0])}; }, {a}, {e}); });
}

TEST_P(MathTest, SubtractScalarInvalidDtype) {
    Array a = Ones({3}, Dtype::kBool);
    EXPECT_THROW(Subtract(a, Scalar{true}), DtypeError);
}

TEST_P(MathTest, SubtractBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(-2).WithPadding(1)).RequireGrad();
    Array b = (*testing::BuildArray(shape).WithData<T>({-6, -4, -2, 2, 4, 6}).WithPadding(2)).RequireGrad();
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(3);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {Subtract(xs[0], xs[1])}; }, {a, b}, {go}, {eps, eps});
}

TEST_P(MathTest, SubtractDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(-2).WithPadding(1)).RequireGrad();
    Array b = (*testing::BuildArray(shape).WithData<T>({-6, -4, -2, 2, 4, 6}).WithPadding(2)).RequireGrad();
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(3)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.3, 0.1).WithPadding(4);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Subtract(xs[0], xs[1]);
                return {y * y};  // to make it nonlinear
            },
            {a, b},
            {go},
            {ggi, ggi},
            {eps, eps, eps});
}

TEST_P(MathTest, SubtractScalarBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Scalar b{T{2.0}};
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-1, Dtype::kFloat64);

    // array - scalar
    CheckBackward([b](const std::vector<Array>& xs) -> std::vector<Array> { return {Subtract(xs[0], b)}; }, {a}, {go}, {eps});
    // scalar - array
    CheckBackward([b](const std::vector<Array>& xs) -> std::vector<Array> { return {Subtract(b, xs[0])}; }, {a}, {go}, {eps});
}

TEST_P(MathTest, SubtractScalarDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Scalar b{T{2.0}};
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.3, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-1, Dtype::kFloat64);

    // array - scalar
    CheckDoubleBackwardComputation(
            [b](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Subtract(xs[0], b);
                return {y * y};  // to make it nonlinear
            },
            {a},
            {go},
            {ggi},
            {eps, eps});
    // scalar - array
    CheckDoubleBackwardComputation(
            [b](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Subtract(b, xs[0]);
                return {y * y};  // to make it nonlinear
            },
            {a},
            {go},
            {ggi},
            {eps, eps});
}

TEST_THREAD_SAFE_P(MathTest, Multiply) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Array b = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Array e = testing::BuildArray({3, 1}).WithData<float>({1, 4, 9});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Multiply(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, MultiplyNonContiguous) {
    Array a = Array(testing::BuildArray({3, 3}).WithLinearData<int32_t>()).At({Slice{}, Slice{1, 2}});
    Array b = FullLike(a, 2);
    Array e = testing::BuildArray({3, 1}).WithData<int32_t>({2, 8, 14});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Multiply(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, MultiplyBroadcast1) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Full({3, 1}, 2, Dtype::kInt32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(0, 2);

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Multiply(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, MultiplyBroadcast2) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Full({3}, 2, Dtype::kInt32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<int32_t>(0, 2);

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Multiply(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, MultiplyBroadcast3) {
    Array a = testing::BuildArray({3}).WithLinearData<int32_t>();
    Array b = Full({3, 3}, 2, Dtype::kInt32);
    Array e = testing::BuildArray({3, 3}).WithData<int32_t>({0, 2, 4, 0, 2, 4, 0, 2, 4});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Multiply(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, MultiplyBroadcast4) {
    Array a = testing::BuildArray({3, 1}).WithLinearData<int32_t>(1);
    Array b = testing::BuildArray({1, 2}).WithLinearData<int32_t>(1);
    Array e = testing::BuildArray({3, 2}).WithData<int32_t>({1, 2, 2, 4, 3, 6});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Multiply(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_P(MathTest, MultiplyInvalidBroadcast) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<int32_t>();
    Array b = Full({4}, 2, Dtype::kInt32);
    EXPECT_THROW(Multiply(a, b), ChainerxError);
}

TEST_THREAD_SAFE_P(MathTest, MultiplyArrayScalar) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Scalar b{2.f};
    Array e = testing::BuildArray({3, 1}).WithData<float>({2, 4, 6});

    Run([&]() { testing::CheckForward([&b](const std::vector<Array>& xs) { return std::vector<Array>{Multiply(xs[0], b)}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(MathTest, MultiplyScalarArray) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Scalar b{2.f};
    Array e = testing::BuildArray({3, 1}).WithData<float>({2, 4, 6});

    Run([&]() { testing::CheckForward([&b](const std::vector<Array>& xs) { return std::vector<Array>{Multiply(b, xs[0])}; }, {a}, {e}); });
}

TEST_P(MathTest, MultiplyBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(-2).WithPadding(1)).RequireGrad();
    Array b = (*testing::BuildArray(shape).WithData<T>({-6, -4, -2, 2, 4, 6}).WithPadding(2)).RequireGrad();
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(3);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {Multiply(xs[0], xs[1])}; }, {a, b}, {go}, {eps, eps});
}

TEST_P(MathTest, MultiplyDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(-2).WithPadding(1)).RequireGrad();
    Array b = (*testing::BuildArray(shape).WithData<T>({-6, -4, -2, 2, 4, 6}).WithPadding(2)).RequireGrad();
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(3)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.3, 0.1).WithPadding(4);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Multiply(xs[0], xs[1]);
                return {y * y};  // to make it nonlinear
            },
            {a, b},
            {go},
            {ggi, ggi},
            {eps, eps, eps});
}

TEST_P(MathTest, MultiplyScalarBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Scalar s{T{2.0}};
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-1, Dtype::kFloat64);

    // array * scalar
    CheckBackward([s](const std::vector<Array>& xs) -> std::vector<Array> { return {Multiply(xs[0], s)}; }, {a}, {go}, {eps});
    // scalar * array
    CheckBackward([s](const std::vector<Array>& xs) -> std::vector<Array> { return {Multiply(s, xs[0])}; }, {a}, {go}, {eps});
}

TEST_P(MathTest, MultiplyScalarDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Scalar s{T{2.0}};
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.3, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-1, Dtype::kFloat64);

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

TEST_THREAD_SAFE_P(MathTest, FloorDivide) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({-3, -3, 0}).WithPadding(1);
    Array b = testing::BuildArray({3, 1}).WithData<float>({2, -2, 1}).WithPadding(2);
    Array e = testing::BuildArray({3, 1}).WithData<float>({-2, 1, 0});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{FloorDivide(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, Divide) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({-3, -3, 0}).WithPadding(1);
    Array b = testing::BuildArray({3, 1}).WithData<float>({2, -2, 1}).WithPadding(2);
    Array e = testing::BuildArray({3, 1}).WithData<float>({-1.5f, 1.5f, 0});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Divide(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, DivideBroadcast1) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<float>().WithPadding(1);
    Array b = Full({3, 1}, 2.0f, Dtype::kFloat32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<float>(0, 0.5f);

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Divide(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, DivideBroadcast2) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<float>().WithPadding(1);
    Array b = Full({3}, 2.0f, Dtype::kFloat32);
    Array e = testing::BuildArray({3, 3}).WithLinearData<float>(0, 0.5f);

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Divide(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, DivideBroadcast3) {
    Array a = testing::BuildArray({3}).WithLinearData<float>().WithPadding(1);
    Array b = Full({3, 3}, 2.0f, Dtype::kFloat32);
    Array e = testing::BuildArray({3, 3}).WithData<float>({0.0f, 0.5f, 1.0f, 0.0f, 0.5f, 1.0f, 0.0f, 0.5f, 1.0f});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Divide(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, DivideBroadcast4) {
    Array a = testing::BuildArray({3, 1}).WithLinearData<float>().WithPadding(1);
    Array b = Full({1, 2}, 2.0f, Dtype::kFloat32);
    Array e = testing::BuildArray({3, 2}).WithData<float>({0.0f, 0.0f, 0.5f, 0.5f, 1.0f, 1.0f});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Divide(xs[0], xs[1])}; }, {a, b}, {e});
    });
}

TEST_P(MathTest, DivideInvalidBroadcast) {
    Array a = testing::BuildArray({3, 3}).WithLinearData<float>();
    Array b = Ones({4}, Dtype::kFloat32);
    EXPECT_THROW(Divide(a, b), ChainerxError);
}

TEST_P(MathTest, DivideScalar) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({1.f, 2.f, 3.f});
    Array e = testing::BuildArray({3, 1}).WithData<float>({0.5f, 1.f, 1.5f});

    testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Divide(xs[0], Scalar{2.f})}; }, {a}, {e});
}

TEST_P(MathTest, DivideInteger) {
    Array a = testing::BuildArray({3, 1}).WithData<int64_t>({1, 2, 3});
    Array b = testing::BuildArray({3, 1}).WithData<int64_t>({2, 2, 12});
    Array e = testing::BuildArray({3, 1}).WithData<double>({0.5f, 1.f, 0.25f});

    testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Divide(xs[0], xs[1])}; }, {a, b}, {e});
}

TEST_P(MathTest, DivideScalarInteger) {
    Array a = testing::BuildArray({3, 1}).WithData<int64_t>({1, 2, 3});
    Array e = testing::BuildArray({3, 1}).WithData<double>({0.5f, 1.f, 1.5f});

    testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Divide(xs[0], Scalar{2})}; }, {a}, {e});
}

TEST_P(MathTest, DivideBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(-2).WithPadding(1)).RequireGrad();
    Array b = (*testing::BuildArray(shape).WithData<T>({-6, -4, -2, 2, 4, 6}).WithPadding(2)).RequireGrad();
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(3);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {Divide(xs[0], xs[1])}; }, {a, b}, {go}, {eps, eps});
}

TEST_P(MathTest, DivideDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(-2).WithPadding(1)).RequireGrad();
    Array b = (*testing::BuildArray(shape).WithData<T>({-6, -4, -2, 2, 4, 6}).WithPadding(2)).RequireGrad();
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(3)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.3, 0.1).WithPadding(4);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {Divide(xs[0], xs[1])}; },
            {a, b},
            {go},
            {ggi, ggi},
            {eps, eps, eps});
}

TEST_P(MathTest, DivideScalarBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Scalar s{T{2.0}};
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-1, Dtype::kFloat64);

    // array / scalar
    CheckBackward([s](const std::vector<Array>& xs) -> std::vector<Array> { return {Divide(xs[0], s)}; }, {a}, {go}, {eps});

    // TODO(hvy): Also test scalar / array, when supported.
}

TEST_P(MathTest, DivideScalarDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Scalar s{T{2.0}};
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.3, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-1, Dtype::kFloat64);

    // array * scalar
    CheckDoubleBackwardComputation(
            [s](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Divide(xs[0], s);
                return {y * y};  // to make it nonlinear
            },
            {a},
            {go},
            {ggi},
            {eps, eps});

    // TODO(hvy): Also test scalar / array, when supported.
}

TEST_THREAD_SAFE_P(MathTest, ChainedMath) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Array b = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Array e = testing::BuildArray({3, 1}).WithData<float>({2, 6, 12});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) { return std::vector<Array>{Add(xs[0], Multiply(xs[0], xs[1]))}; }, {a, b}, {e});
    });
}

TEST_P(MathTest, ChainedInplaceMath) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Array b = testing::BuildArray({3, 1}).WithData<float>({1, 2, 3});
    Array e = testing::BuildArray({3, 1}).WithData<float>({2, 6, 12});
    internal::IMultiply(b, a);
    internal::IAdd(a, b);
    EXPECT_ARRAY_EQ(e, a);
}

TEST_THREAD_SAFE_P(MathTest, Reciprocal) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({-1.f, 2.f, -.2f});
    Array e = testing::BuildArray({3, 1}).WithData<float>({-1.f / 1.f, 1.f / 2.f, -1.f / .2f});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Reciprocal(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(MathTest, ReciprocalZero) {
    Array a = testing::BuildArray({1}).WithData<float>({0.f});
    Array e = testing::BuildArray({1}).WithData<float>({std::numeric_limits<float>::infinity()});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Reciprocal(xs[0])}; }, {a}, {e}); });
}

TEST_P(MathTest, ReciprocalBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(1.).WithPadding(1)).RequireGrad();
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {Reciprocal(xs[0])}; }, {a}, {go}, {eps});
}

TEST_P(MathTest, ReciprocalDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(1.).WithPadding(1)).RequireGrad();
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.3, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {Reciprocal(xs[0])}; }, {a}, {go}, {ggi}, {eps, eps});
}

TEST_THREAD_SAFE_P(MathTest, Sum) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1);
    Array e = testing::BuildArray({2}).WithData<T>({630.0f, 1926.0f});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Sum(xs[0], Axes{2, 1, -1})}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, SumAllAxes) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 3}).WithLinearData<T>().WithPadding(1);
    Array e = testing::BuildArray({}).WithData<T>({153.0f});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Sum(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(MathTest, SumZero) {
    using T = float;

    Array a = testing::BuildArray({0}).WithData<T>({});
    Array e = testing::BuildArray({}).WithData<T>({0.0f});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Sum(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(MathTest, SumOne) {
    using T = float;

    Array a = testing::BuildArray({}).WithData<T>({42.0f}).WithPadding(1);
    Array e = testing::BuildArray({}).WithData<T>({42.0f});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Sum(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(MathTest, SumTwo) {
    using T = float;

    Array a = testing::BuildArray({2}).WithData<T>({42.0f, 37.0f}).WithPadding(1);
    Array e = testing::BuildArray({}).WithData<T>({79.0f});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Sum(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(MathTest, SumLarge) {
    using T = int64_t;

    Array a = testing::BuildArray({0x100000}).WithLinearData<T>().WithPadding(1);
    Array e = testing::BuildArray({}).WithData<T>({0x7ffff80000});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Sum(xs[0], Axes{0})}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(MathTest, SumKeepDims) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 2, 4}).WithLinearData<T>().WithPadding(1);
    Array e = testing::BuildArray({2, 1, 2, 1}).WithData<T>({114.0f, 162.0f, 402.0f, 450.0f});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    Array y = Sum(xs[0], Axes{-1, 1}, true);
                    EXPECT_EQ(0, y.strides()[1]);
                    EXPECT_EQ(0, y.strides()[3]);
                    return std::vector<Array>{y};
                },
                {a},
                {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, SumSignedInt) {
    using T = int8_t;

    Array a = testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1);
    Array e = testing::BuildArray({2}).WithData<int64_t>({630, 1926});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Sum(xs[0], Axes{2, 1, -1})}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, SumUnsignedInt) {
    // TODO(niboshi): The resulted dtype should be uint64 instead of int64.
    using T = uint8_t;

    Array a = testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1);
    Array e = testing::BuildArray({2}).WithData<int64_t>({630, 1926});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Sum(xs[0], Axes{2, 1, -1})}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, SumBool) {
    using T = bool;

    Array a = testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1);
    Array e = testing::BuildArray({2}).WithData<int64_t>({35, 36});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Sum(xs[0], Axes{2, 1, -1})}; }, {a}, {e});
    });
}

TEST_P(MathTest, InvalidSumDuplicateAxes) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<T>();
    EXPECT_THROW(Sum(a, Axes{1, 1}), ChainerxError);
}

TEST_P(MathTest, InvalidSumOutOfRangeAxes) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<T>();
    EXPECT_THROW(Sum(a, Axes{3}), DimensionError);
}

TEST_P(MathTest, SumBackward) {
    using T = double;

    CheckBackward(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                return {Sum(xs[0], Axes{1, 3})};
            },
            {(*testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {testing::BuildArray({2, 4}).WithLinearData<T>(-0.1, 0.1)},
            {Full({2, 3, 4, 3}, 1e-1, Dtype::kFloat64)});
}

TEST_P(MathTest, SumDoubleBackward_Keepdims) {
    using T = double;

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Sum(xs[0], Axes{1, 3}, true);
                return {y * y};  // to make it nonlinear
            },
            {(*testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {(*testing::BuildArray({2, 1, 4, 1}).WithLinearData<T>(-0.1, 0.1)).RequireGrad()},
            {testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>()},
            {Full({2, 3, 4, 3}, 1e-1, Dtype::kFloat64), Full({2, 1, 4, 1}, 1e-1, Dtype::kFloat64)});
}

TEST_P(MathTest, SumDoubleBackward_NoKeepdims) {
    using T = double;

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Sum(xs[0], Axes{1, 3}, false);
                return {y * y};  // to make it nonlinear
            },
            {(*testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {(*testing::BuildArray({2, 4}).WithLinearData<T>(-0.1, 0.1)).RequireGrad()},
            {testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>()},
            {Full({2, 3, 4, 3}, 1e-1, Dtype::kFloat64), Full({2, 4}, 1e-1, Dtype::kFloat64)});
}

TEST_THREAD_SAFE_P(MathTest, AMax) {
    Array a = testing::BuildArray({2, 3, 4, 3}).WithLinearData<float>().WithPadding(1);
    Array e = testing::BuildArray({3}).WithData<float>({47.f, 59.f, 71.f});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{AMax(xs[0], Axes{2, 0, -1})}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, AMaxAllAxes) {
    Array a = testing::BuildArray({2, 3, 3}).WithLinearData<float>().WithPadding(1);
    Array e = testing::BuildArray({}).WithData<float>({17.f});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{AMax(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(MathTest, AMaxZeroSized) {
    Array a = Empty({0, 2}, Dtype::kFloat32);
    Array e = testing::BuildArray({0}).WithData<float>({});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{AMax(xs[0], Axes{1})}; }, {a}, {e}); });
}

TEST_P(MathTest, AMaxAlongZeroSized) {
    Array a = Empty({0, 2}, Dtype::kFloat32);
    EXPECT_THROW(AMax(a, Axes{0}), DimensionError);
    EXPECT_THROW(AMax(a), DimensionError);
}

TEST_P(MathTest, AMaxBackward) {
    using T = double;

    CheckBackward(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                return {AMax(xs[0], Axes{1, 3})};
            },
            {(*testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {testing::BuildArray({2, 4}).WithLinearData<T>(-0.1, 0.1)},
            {Full({2, 3, 4, 3}, 1e-1, Dtype::kFloat64)});
}

TEST_P(MathTest, AMaxDoubleBackward_Keepdims) {
    using T = double;

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = AMax(xs[0], Axes{1, 3}, true);
                return {y * y};  // to make it nonlinear
            },
            {(*testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {(*testing::BuildArray({2, 1, 4, 1}).WithLinearData<T>(-0.1, 0.1)).RequireGrad()},
            {testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>()},
            {Full({2, 3, 4, 3}, 1e-1, Dtype::kFloat64), Full({2, 1, 4, 1}, 1e-1, Dtype::kFloat64)});
}

TEST_P(MathTest, AMaxDoubleBackward_NoKeepdims) {
    using T = double;

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = AMax(xs[0], Axes{1, 3}, false);
                return {y * y};  // to make it nonlinear
            },
            {(*testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {(*testing::BuildArray({2, 4}).WithLinearData<T>(-0.1, 0.1)).RequireGrad()},
            {testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>()},
            {Full({2, 3, 4, 3}, 1e-1, Dtype::kFloat64), Full({2, 4}, 1e-1, Dtype::kFloat64)});
}

TEST_THREAD_SAFE_P(MathTest, MaximumArrayScalar) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({-1.f, 2.f, -.2f});
    Array e = testing::BuildArray({3, 1}).WithData<float>({0.f, 2.f, 0.f});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Maximum(xs[0], Scalar{0.f})}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, MaximumScalarArray) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({-1.f, 2.f, -.2f});
    Array e = testing::BuildArray({3, 1}).WithData<float>({0.f, 2.f, 0.f});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Maximum(Scalar{0.f}, xs[0])}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, MaximumScalarEmpty) {
    Array a = testing::BuildArray({0}).WithData<float>({});
    Array e = testing::BuildArray({0}).WithData<float>({});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Maximum(xs[0], Scalar{0.f})}; }, {a}, {e});
    });
}

TEST_P(MathTest, MaximumScalarBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Scalar s{T{0.2}};
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-1, Dtype::kFloat64);

    // Maximum(array, scalar)
    CheckBackward([s](const std::vector<Array>& xs) -> std::vector<Array> { return {Maximum(xs[0], s)}; }, {a}, {go}, {eps});
    // Maximum(scalar, array)
    CheckBackward([s](const std::vector<Array>& xs) -> std::vector<Array> { return {Maximum(s, xs[0])}; }, {a}, {go}, {eps});
}

TEST_P(MathTest, MaximumScalarDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Scalar s{T{0.2}};
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.3, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-1, Dtype::kFloat64);

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

TEST_THREAD_SAFE_P(MathTest, MinimumArrayScalar) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({-1.f, 2.f, -.2f});
    Array e = testing::BuildArray({3, 1}).WithData<float>({-1.f, 0.f, -.2f});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Minimum(xs[0], Scalar{0.f})}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, MinimumScalarArray) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({-1.f, 2.f, -.2f});
    Array e = testing::BuildArray({3, 1}).WithData<float>({-1.f, 0.f, -.2f});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Minimum(Scalar{0.f}, xs[0])}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, MinimumScalarEmpty) {
    Array a = testing::BuildArray({0}).WithData<float>({});
    Array e = testing::BuildArray({0}).WithData<float>({});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Minimum(xs[0], Scalar{0.f})}; }, {a}, {e});
    });
}

TEST_P(MathTest, MinimumScalarBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Scalar s{T{0.2}};
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-1, Dtype::kFloat64);

    // Minimum(array, scalar)
    CheckBackward([s](const std::vector<Array>& xs) -> std::vector<Array> { return {Minimum(xs[0], s)}; }, {a}, {go}, {eps});
    // Minimum(scalar, array)
    CheckBackward([s](const std::vector<Array>& xs) -> std::vector<Array> { return {Minimum(s, xs[0])}; }, {a}, {go}, {eps});
}

TEST_P(MathTest, MinimumScalarDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Scalar s{T{0.2}};
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.3, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-1, Dtype::kFloat64);

    // Minimum(array, scalar)
    CheckDoubleBackwardComputation(
            [s](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Minimum(xs[0], s);
                return {y * y};  // to make it nonlinear
            },
            {a},
            {go},
            {ggi},
            {eps, eps});
    // Minimum(scalar, array)
    CheckDoubleBackwardComputation(
            [s](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Minimum(s, xs[0]);
                return {y * y};  // to make it nonlinear
            },
            {a},
            {go},
            {ggi},
            {eps, eps});
}

TEST_THREAD_SAFE_P(MathTest, Exp) {
    Array a = testing::BuildArray({5}).WithData<float>(
            {0.f, 1.f, std::log(3.f), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()});
    Array e = testing::BuildArray({5}).WithData<float>({1.f, std::exp(1.f), 3.f, std::numeric_limits<float>::infinity(), 0});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Exp(xs[0])}; }, {a}, {e}); });
}

TEST_P(MathTest, ExpBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {Exp(xs[0])}; }, {a}, {go}, {eps});
}

TEST_P(MathTest, ExpDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {Exp(xs[0])}; }, {a}, {go}, {ggi}, {eps, eps});
}

TEST_THREAD_SAFE_P(MathTest, Log) {
    Array a = testing::BuildArray({6}).WithData<float>({0.0f, 1.0f, 3.0f, -1.f, std::exp(-4.0f), std::exp(4.0f)}).WithPadding(1);
    Array e = testing::BuildArray({6}).WithData<float>(
            {-std::numeric_limits<float>::infinity(), 0.0f, std::log(3.0f), std::nanf(""), -4.0f, 4.0f});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Log(xs[0])}; }, {a}, {e}); });
}

TEST_P(MathTest, LogBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(1.0, 1.0).WithPadding(1)).RequireGrad();
    Array go = testing::BuildArray(shape).WithLinearData<T>(0.1, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {Log(xs[0])}; }, {a}, {go}, {eps});
}

TEST_P(MathTest, LogDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(1.0, 1.0).WithPadding(1)).RequireGrad();
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(0.1, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {Log(xs[0])}; }, {a}, {go}, {ggi}, {eps, eps});
}

TEST_THREAD_SAFE_P(MathTest, LogSumExp) {
    using T = double;
    std::vector<T> adata{-1, 0, 1, 2, 3, 4};
    Array a = testing::BuildArray({2, 3}).WithData<T>(adata).WithPadding(1);
    Array e = testing::BuildArray({}).WithData<T>({std::log(
            std::exp(adata[0]) + std::exp(adata[1]) + std::exp(adata[2]) + std::exp(adata[3]) + std::exp(adata[4]) + std::exp(adata[5]))});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{LogSumExp(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(MathTest, LogSumExpReduceFirstAxis) {
    using T = double;
    std::vector<T> adata{-1, 0, 1, 2, 3, 4};
    Array a = testing::BuildArray({2, 3}).WithData<T>(adata).WithPadding(1);
    Array e = testing::BuildArray({3}).WithData<T>({std::log(std::exp(adata[0]) + std::exp(adata[3])),
                                                    std::log(std::exp(adata[1]) + std::exp(adata[4])),
                                                    std::log(std::exp(adata[2]) + std::exp(adata[5]))});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{LogSumExp(xs[0], Axes{0})}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, LogSumExpReduceSecondAxis) {
    using T = double;
    std::vector<T> adata{-1, 0, 1, 2, 3, 4};
    Array a = testing::BuildArray({2, 3}).WithData<T>(adata).WithPadding(1);
    Array e = testing::BuildArray({2}).WithData<T>({std::log(std::exp(adata[0]) + std::exp(adata[1]) + std::exp(adata[2])),
                                                    std::log(std::exp(adata[3]) + std::exp(adata[4]) + std::exp(adata[5]))});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{LogSumExp(xs[0], Axes{1})}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, LogSumExpReduceMultipleAxes) {
    using T = double;
    std::vector<T> adata{-1, 0, 1, 2, 3, 4};
    Array a = testing::BuildArray({1, 3, 1, 2}).WithData<T>(adata).WithPadding(1);
    Array e = testing::BuildArray({3}).WithData<T>({std::log(std::exp(adata[0]) + std::exp(adata[1])),
                                                    std::log(std::exp(adata[2]) + std::exp(adata[3])),
                                                    std::log(std::exp(adata[4]) + std::exp(adata[5]))});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{LogSumExp(xs[0], Axes{0, 2, 3})}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, LogSumExpKeepdims) {
    using T = double;
    std::vector<T> adata{-1, 0, 1, 2, 3, 4};
    Array a = testing::BuildArray({2, 3}).WithData<T>(adata).WithPadding(1);
    Array e = testing::BuildArray({2, 1}).WithData<T>({std::log(std::exp(adata[0]) + std::exp(adata[1]) + std::exp(adata[2])),
                                                       std::log(std::exp(adata[3]) + std::exp(adata[4]) + std::exp(adata[5]))});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{LogSumExp(xs[0], Axes{1}, true)}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, LogSumExpReduceMultipleAxesKeepdims) {
    using T = double;
    std::vector<T> adata{-1, 0, 1, 2, 3, 4};
    Array a = testing::BuildArray({2, 3}).WithData<T>(adata).WithPadding(1);
    Array e = testing::BuildArray({1, 1}).WithData<T>({std::log(
            std::exp(adata[0]) + std::exp(adata[1]) + std::exp(adata[2]) + std::exp(adata[3]) + std::exp(adata[4]) + std::exp(adata[5]))});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    return std::vector<Array>{LogSumExp(xs[0], Axes{0, 1}, true)};
                },
                {a},
                {e});
    });
}

TEST_P(MathTest, LogSumExpBackward) {
    using T = double;
    Array a = (*testing::BuildArray({2, 3}).WithLinearData<T>(-3).WithPadding(1)).RequireGrad();
    Array go = testing::BuildArray({}).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full({2, 3}, 1e-3, Dtype::kFloat64);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {LogSumExp(xs[0])}; }, {a}, {go}, {eps});
}

TEST_P(MathTest, LogSumExpDoubleBackward) {
    using T = double;
    Array a = (*testing::BuildArray({2, 3}).WithLinearData<T>(-3).WithPadding(1)).RequireGrad();
    Array go = (*testing::BuildArray({}).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array gga = testing::BuildArray({2, 3}).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps_a = Full({2, 3}, 1e-3, Dtype::kFloat64);
    Array eps_go = Full({}, 1e-3, Dtype::kFloat64);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {LogSumExp(xs[0])}; }, {a}, {go}, {gga}, {eps_a, eps_go});
}

TEST_THREAD_SAFE_P(MathTest, LogSoftmax) {
    using T = double;
    Shape shape{2, 3};
    std::vector<T> adata{-1, 0, 1, 2, 3, 4};
    std::vector<T> log_z{std::log(std::exp(adata[0]) + std::exp(adata[1]) + std::exp(adata[2])),
                         std::log(std::exp(adata[3]) + std::exp(adata[4]) + std::exp(adata[5]))};
    Array a = testing::BuildArray(shape).WithData<T>(adata).WithPadding(1);
    Array e = testing::BuildArray(shape).WithData<T>(
            {adata[0] - log_z[0], adata[1] - log_z[0], adata[2] - log_z[0], adata[3] - log_z[1], adata[4] - log_z[1], adata[5] - log_z[1]});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{LogSoftmax(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(MathTest, LogSoftmaxAlongFirstAxis) {
    using T = double;
    Shape shape{2, 3};
    std::vector<T> adata{-1, 0, 1, 2, 3, 4};
    std::vector<T> log_z{std::log(std::exp(adata[0]) + std::exp(adata[3])),
                         std::log(std::exp(adata[1]) + std::exp(adata[4])),
                         std::log(std::exp(adata[2]) + std::exp(adata[5]))};
    Array a = testing::BuildArray(shape).WithData<T>(adata).WithPadding(1);
    Array e = testing::BuildArray(shape).WithData<T>(
            {adata[0] - log_z[0], adata[1] - log_z[1], adata[2] - log_z[2], adata[3] - log_z[0], adata[4] - log_z[1], adata[5] - log_z[2]});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{LogSoftmax(xs[0], Axes{0})}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, LogSoftmaxAlongSecondAxis) {
    using T = double;
    Shape shape{2, 3};
    std::vector<T> adata{-1, 0, 1, 2, 3, 4};
    std::vector<T> log_z{std::log(std::exp(adata[0]) + std::exp(adata[1]) + std::exp(adata[2])),
                         std::log(std::exp(adata[3]) + std::exp(adata[4]) + std::exp(adata[5]))};
    Array a = testing::BuildArray(shape).WithData<T>(adata).WithPadding(1);
    Array e = testing::BuildArray(shape).WithData<T>(
            {adata[0] - log_z[0], adata[1] - log_z[0], adata[2] - log_z[0], adata[3] - log_z[1], adata[4] - log_z[1], adata[5] - log_z[1]});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{LogSoftmax(xs[0], Axes{1})}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, LogSoftmaxAlongMultipleAxes) {
    using T = double;
    Shape shape{2, 3};
    std::vector<T> adata{-1, 0, 1, 2, 3, 4};
    T log_z = std::log(
            std::exp(adata[0]) + std::exp(adata[1]) + std::exp(adata[2]) + std::exp(adata[3]) + std::exp(adata[4]) + std::exp(adata[5]));
    Array a = testing::BuildArray(shape).WithData<T>(adata).WithPadding(1);
    Array e = testing::BuildArray(shape).WithData<T>(
            {adata[0] - log_z, adata[1] - log_z, adata[2] - log_z, adata[3] - log_z, adata[4] - log_z, adata[5] - log_z});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{LogSoftmax(xs[0], Axes{0, 1})}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(MathTest, LogSoftmaxHighDimAlongDefaultSecondAxis) {
    using T = double;
    Shape shape{1, 3, 1, 2};
    std::vector<T> adata{-1, 0, 1, 2, 3, 4};
    std::vector<T> log_z{std::log(std::exp(adata[0]) + std::exp(adata[2]) + std::exp(adata[4])),
                         std::log(std::exp(adata[1]) + std::exp(adata[3]) + std::exp(adata[5]))};
    Array a = testing::BuildArray(shape).WithData<T>(adata).WithPadding(1);
    Array e = testing::BuildArray(shape).WithData<T>(
            {adata[0] - log_z[0], adata[1] - log_z[1], adata[2] - log_z[0], adata[3] - log_z[1], adata[4] - log_z[0], adata[5] - log_z[1]});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{LogSoftmax(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(MathTest, LogSoftmaxHighDimAlongSecondAxis) {
    using T = double;
    Shape shape{1, 3, 1, 2};
    std::vector<T> adata{-1, 0, 1, 2, 3, 4};
    std::vector<T> log_z{std::log(std::exp(adata[0]) + std::exp(adata[2]) + std::exp(adata[4])),
                         std::log(std::exp(adata[1]) + std::exp(adata[3]) + std::exp(adata[5]))};
    Array a = testing::BuildArray(shape).WithData<T>(adata).WithPadding(1);
    Array e = testing::BuildArray(shape).WithData<T>(
            {adata[0] - log_z[0], adata[1] - log_z[1], adata[2] - log_z[0], adata[3] - log_z[1], adata[4] - log_z[0], adata[5] - log_z[1]});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{LogSoftmax(xs[0], Axes{1})}; }, {a}, {e});
    });
}

TEST_P(MathTest, LogSoftmaxBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(-3).WithPadding(1)).RequireGrad();
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {LogSoftmax(xs[0])}; }, {a}, {go}, {eps});
}

TEST_P(MathTest, LogSoftmaxDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(-3).WithPadding(1)).RequireGrad();
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {LogSoftmax(xs[0])}; }, {a}, {go}, {ggi}, {eps, eps});
}

TEST_THREAD_SAFE_P(MathTest, Sqrt) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({-1.f, 2.f, 0.f});
    Array e = testing::BuildArray({3, 1}).WithData<float>({std::sqrt(-1.f), std::sqrt(2.f), std::sqrt(0.f)});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Sqrt(xs[0])}; }, {a}, {e}); });
}

TEST_P(MathTest, SqrtBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(1).WithPadding(1)).RequireGrad();
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {Sqrt(xs[0])}; }, {a}, {go}, {eps});
}

TEST_P(MathTest, SqrtDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(1).WithPadding(1)).RequireGrad();
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.3, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {Sqrt(xs[0])}; }, {a}, {go}, {ggi}, {eps, eps});
}

TEST_THREAD_SAFE_P(MathTest, Tanh) {
    Array a = testing::BuildArray({3, 1}).WithData<float>({-1.f, 2.f, 0.f});
    Array e = testing::BuildArray({3, 1}).WithData<float>({std::tanh(-1.f), std::tanh(2.f), std::tanh(0.f)});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Tanh(xs[0])}; }, {a}, {e}); });
}

TEST_P(MathTest, TanhBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(1).WithPadding(1)).RequireGrad();
    Array go = testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {Tanh(xs[0])}; }, {a}, {go}, {eps});
}

TEST_P(MathTest, TanhDoubleBackward) {
    using T = double;
    Shape shape{2, 3};
    Array a = (*testing::BuildArray(shape).WithLinearData<T>(1).WithPadding(1)).RequireGrad();
    Array go = (*testing::BuildArray(shape).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggi = testing::BuildArray(shape).WithLinearData<T>(-0.3, 0.1).WithPadding(1);
    Array eps = Full(shape, 1e-3, Dtype::kFloat64);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {Tanh(xs[0])}; }, {a}, {go}, {ggi}, {eps, eps});
}

TEST_THREAD_SAFE_P(MathTest, IsNan) {
    Array a = testing::BuildArray({5, 1}).WithData<float>(
            {-1.f, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), std::nanf(""), std::nanf("0xf")});
    Array e = testing::BuildArray({5, 1}).WithData<bool>({false, false, false, true, true});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{IsNan(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(MathTest, IsInf) {
    Array a = testing::BuildArray({5, 1}).WithData<float>(
            {-1.f, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), std::nanf(""), std::nanf("0xf")});
    Array e = testing::BuildArray({5, 1}).WithData<bool>({false, true, true, false, false});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{IsInf(xs[0])}; }, {a}, {e}); });
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        MathTest,
        ::testing::Values(
#ifdef CHAINERX_ENABLE_CUDA
                std::string{"cuda"},
#endif  // CHAINERX_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace chainerx
