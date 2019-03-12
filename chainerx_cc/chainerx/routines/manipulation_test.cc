#include "chainerx/routines/manipulation.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/array_index.h"
#include "chainerx/axes.h"
#include "chainerx/backward.h"
#include "chainerx/check_backward.h"
#include "chainerx/device_id.h"
#include "chainerx/error.h"
#include "chainerx/shape.h"
#include "chainerx/strides.h"
#include "chainerx/testing/array.h"
#include "chainerx/testing/array_check.h"
#include "chainerx/testing/device_session.h"
#include "chainerx/testing/routines.h"
#include "chainerx/testing/threading.h"

namespace chainerx {
namespace {

class ManipulationTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        const std::string& backend_name = GetParam();
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

private:
    nonstd::optional<testing::DeviceSession> device_session_;
};

TEST_THREAD_SAFE_P(ManipulationTest, AsScalar) {
    using T = float;
    T value = 2.0f;
    Array a = testing::BuildArray({1, 1, 1}).WithData<T>({value}).WithPadding(1);

    Run([&a, &value]() {
        Scalar s = AsScalar(a);
        EXPECT_EQ(s.kind(), GetKind(TypeToDtype<T>));
        EXPECT_EQ(static_cast<T>(s), value);
    });
}

TEST_P(ManipulationTest, AsScalarInvalidZeroElement) {
    Array a = testing::BuildArray({0}).WithData<float>({});
    EXPECT_THROW(AsScalar(a), DimensionError);
}

TEST_P(ManipulationTest, AsScalarInvalidMoreThanOneElements) {
    Array a = testing::BuildArray({2}).WithData<float>({1.0f, 2.0f});
    EXPECT_THROW(AsScalar(a), DimensionError);
}

TEST_THREAD_SAFE_P(ManipulationTest, RollAxis) {
    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<int32_t>();
    Array e = testing::BuildArray({3, 2, 4}).WithData<int32_t>(
            {0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{RollAxis(xs[0], 1)}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(ManipulationTest, RollAxisWithStart) {
    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<int32_t>();
    Array e = testing::BuildArray({3, 2, 4}).WithData<int32_t>(
            {0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{RollAxis(xs[0], -3, -1)}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(ManipulationTest, Transpose) {
    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<int32_t>();
    Array e = testing::BuildArray({4, 2, 3}).WithData<int32_t>(
            {0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    Array y = Transpose(xs[0], {2, 0, 1});
                    EXPECT_EQ(Strides({4, 48, 16}), y.strides());
                    return std::vector<Array>{y};
                },
                {a},
                {e});
    });
}

TEST_THREAD_SAFE_P(ManipulationTest, TransposeDefaultAxes) {
    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<int32_t>();
    Array e = testing::BuildArray({4, 3, 2}).WithData<int32_t>(
            {0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    Array y = Transpose(xs[0]);
                    EXPECT_EQ(Strides({4, 16, 48}), y.strides());
                    return std::vector<Array>{y};
                },
                {a},
                {e});
    });
}

TEST_THREAD_SAFE_P(ManipulationTest, TransposeNoncontiguous) {
    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<int32_t>().WithPadding(1);
    Array e = testing::BuildArray({4, 2, 3}).WithData<int32_t>(
            {0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Transpose(xs[0], {2, 0, 1})}; }, {a}, {e});
    });
}

TEST_P(ManipulationTest, TransposeBackward) {
    CheckBackward(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                return {Transpose(xs[0], {2, 0, 1})};
            },
            {(*testing::BuildArray({2, 3, 4}).WithLinearData<float>()).RequireGrad()},
            {testing::BuildArray({4, 2, 3}).WithLinearData<float>(-1.f, 0.1f)},
            {Full({2, 3, 4}, 1e-2f)});
}

TEST_P(ManipulationTest, TransposeDoubleBackward) {
    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto t = Transpose(xs[0], {2, 0, 1});
                return {t * t};  // to make it nonlinear
            },
            {(*testing::BuildArray({2, 3, 4}).WithLinearData<float>()).RequireGrad()},
            {Ones({4, 2, 3}, Dtype::kFloat32).RequireGrad()},
            {Ones({2, 3, 4}, Dtype::kFloat32)},
            {Full({2, 3, 4}, 0.01f), Full({4, 2, 3}, 0.01f)});
}

TEST_THREAD_SAFE_P(ManipulationTest, Reshape) {
    using T = int32_t;
    Shape input_shape{2, 3, 4};
    Shape output_shape{3, 4, 2};

    Array a = testing::BuildArray(input_shape).WithLinearData<T>();
    Array e = testing::BuildArray(output_shape).WithLinearData<T>();

    Run([&]() {
        testing::CheckForward(
                [&output_shape](const std::vector<Array>& xs) {
                    Array y = Reshape(xs[0], output_shape);
                    EXPECT_EQ(xs[0].data().get(), y.data().get()) << "Reshape must be done without copying data";
                    return std::vector<Array>{y};
                },
                {a},
                {e});
    });
}

// #461
TEST_THREAD_SAFE_P(ManipulationTest, ReshapeWithStrideOne) {
    using T = bool;
    Shape input_shape{6};
    Shape output_shape{2, 3};

    Array a = testing::BuildArray(input_shape).WithLinearData<T>();
    Array e = testing::BuildArray(output_shape).WithLinearData<T>();

    Run([&]() {
        testing::CheckForward(
                [&output_shape](const std::vector<Array>& xs) {
                    Array y = Reshape(xs[0], output_shape);
                    EXPECT_EQ(xs[0].data().get(), y.data().get()) << "Reshape must be done without copying data";
                    return std::vector<Array>{y};
                },
                {a},
                {e});
    });
}

// #461
TEST_THREAD_SAFE_P(ManipulationTest, ReshapeNewAxisAtEnd) {
    using T = double;
    Shape input_shape{2, 4};
    Shape output_shape{2, 1, 4, 1};

    Array a = testing::BuildArray(input_shape).WithLinearData<T>();
    Array e = testing::BuildArray(output_shape).WithLinearData<T>();

    Run([&]() {
        testing::CheckForward(
                [&output_shape](const std::vector<Array>& xs) {
                    Array y = Reshape(xs[0], output_shape);
                    EXPECT_EQ(xs[0].data().get(), y.data().get()) << "Reshape must be done without copying data";
                    return std::vector<Array>{y};
                },
                {a},
                {e});
    });
}

// If an input array has a unit-length axis with 0-stride, that axis should not give rise to any copies.
TEST_THREAD_SAFE_P(ManipulationTest, ReshapeNoCopyZeroStrideAxis) {
    using T = int32_t;
    Shape input_shape_before_newaxis{2, 3, 4};
    Shape output_shape{3, 4, 2};

    // The shape of the input array is (2, 1, 3, 4) with strides (48, 0, 16, 4).
    Array a = (*testing::BuildArray(input_shape_before_newaxis).WithLinearData<T>()).At({Slice{}, NewAxis{}, Slice{}, Slice{}});
    ASSERT_TRUE(std::find(a.strides().begin(), a.strides().end(), 0) != a.strides().end());
    Array e = testing::BuildArray(output_shape).WithLinearData<T>();

    Run([&]() {
        testing::CheckForward(
                [&output_shape](const std::vector<Array>& xs) {
                    Array y = Reshape(xs[0], output_shape);
                    EXPECT_EQ(xs[0].data().get(), y.data().get()) << "Reshape must be done without copying data";
                    return std::vector<Array>{y};
                },
                {a},
                {e});
    });
}

TEST_THREAD_SAFE_P(ManipulationTest, ReshapeWithCopy) {
    using T = int32_t;
    Shape input_shape{2, 3, 4};
    Shape output_shape{2, 12};

    Array a = testing::BuildArray(input_shape).WithLinearData<T>().WithPadding(1);
    Array e = testing::BuildArray(output_shape).WithLinearData<T>();

    Run([&]() {
        testing::CheckForward(
                [&output_shape](const std::vector<Array>& xs) {
                    Array y = Reshape(xs[0], output_shape);
                    EXPECT_NE(xs[0].data().get(), y.data().get()) << "Reshape must be done with copy";
                    return std::vector<Array>{y};
                },
                {a},
                {e});
    });
}

TEST_THREAD_SAFE_P(ManipulationTest, ReshapeInferDimension) {
    using T = int32_t;
    Shape input_shape{2, 3, 4};
    Shape output_shape{3, -1, 1};
    Shape inferred_output_shape{3, 8, 1};

    Array a = testing::BuildArray(input_shape).WithLinearData<T>();
    Array e = testing::BuildArray(inferred_output_shape).WithLinearData<T>();

    Run([&]() {
        testing::CheckForward(
                [&output_shape](const std::vector<Array>& xs) {
                    Array y = Reshape(xs[0], output_shape);
                    EXPECT_EQ(xs[0].data().get(), y.data().get()) << "Reshape must be done without copying data";
                    return std::vector<Array>{y};
                },
                {a},
                {e});
    });
}

TEST_P(ManipulationTest, InvalidReshapeTotalSizeMismatch) {
    using T = int32_t;
    Shape input_shape{2, 3, 4};
    Shape output_shape{2, 4, 4};

    Array a = testing::BuildArray(input_shape).WithLinearData<T>();
    EXPECT_THROW(Reshape(a, output_shape), DimensionError);
}

TEST_P(ManipulationTest, InvalidReshapeCannotInfer) {
    using T = int32_t;
    Shape input_shape{2, 3, 4};
    Shape output_shape{2, -1, -1};

    Array a = testing::BuildArray(input_shape).WithLinearData<T>();
    EXPECT_THROW(Reshape(a, output_shape), DimensionError);
}

TEST_THREAD_SAFE_P(ManipulationTest, SqueezeAllUnitLengthAxes) {
    using T = int32_t;

    Array a = testing::BuildArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<T>();
    Array e = testing::BuildArray({2, 3, 4}).WithLinearData<T>();

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Squeeze(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(ManipulationTest, SqueezeSpecifiedUnitLenghtAxes) {
    using T = int32_t;

    Array a = testing::BuildArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<T>();
    Array e = testing::BuildArray({2, 3, 1, 4}).WithLinearData<T>();

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Squeeze(xs[0], Axes{2, 0, 4})}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(ManipulationTest, SqueezeAllAxes) {
    using T = int32_t;

    Array a = testing::BuildArray({1, 1, 1}).WithLinearData<T>();
    Array e = testing::BuildArray({}).WithData<T>({0});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Squeeze(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(ManipulationTest, SqueezeMultipleCalls) {
    using T = int32_t;

    Array a = testing::BuildArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<T>();
    Array e = testing::BuildArray({2, 3, 1, 4}).WithLinearData<T>();

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    return std::vector<Array>{Squeeze(Squeeze(xs[0], Axes{0, 2}), Axes{3})};
                },
                {a},
                {e});
    });
}

TEST_THREAD_SAFE_P(ManipulationTest, SqueezeNonContiguous) {
    using T = int32_t;

    Array a = testing::BuildArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<T>().WithPadding(1);
    Array e = testing::BuildArray({2, 3, 1, 4}).WithLinearData<T>();

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Squeeze(xs[0], Axes{0, 2, 4})}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(ManipulationTest, SqueezeNegativeAxis) {
    using T = int32_t;

    Array a = testing::BuildArray({2, 3, 4, 1}).WithLinearData<T>();
    Array e = testing::BuildArray({2, 3, 4}).WithLinearData<T>();

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Squeeze(xs[0], Axes{-1})}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(ManipulationTest, SqueezeNoSqueezableAxes) {
    using T = int32_t;

    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<T>();

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    Array y = Squeeze(xs[0]);
                    EXPECT_EQ(internal::GetArrayBody(y), internal::GetArrayBody(xs[0]));
                    return std::vector<Array>{y};
                },
                {a},
                {a});
    });
}

TEST_P(ManipulationTest, InvalidSqueezeNonUnitLengthAxis) {
    using T = int32_t;

    Array a = testing::BuildArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<T>();
    EXPECT_THROW(Array b = Squeeze(a, Axes{1}), DimensionError);
}

TEST_P(ManipulationTest, InvalidSqueezeDuplicateAxes) {
    using T = int32_t;

    Array a = testing::BuildArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<T>();
    EXPECT_THROW(Squeeze(a, Axes{0, 2, 2}), ChainerxError);
}

TEST_P(ManipulationTest, InvalidSqueezeOutOfRangeAxes) {
    using T = int32_t;

    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<T>();
    EXPECT_THROW(Squeeze(a, Axes{3}), DimensionError);
}

TEST_P(ManipulationTest, SqueezeBackward) {
    CheckBackward(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                return {Squeeze(xs[0], Axes{0, 2, 4})};
            },
            {(*testing::BuildArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<float>().WithPadding(1)).RequireGrad()},
            {testing::BuildArray({2, 3, 1, 4}).WithLinearData<float>(0.f, 0.1f)},
            {Full({1, 2, 1, 3, 1, 1, 4}, 1e-2f)});
}

TEST_P(ManipulationTest, SqueezeDoubleBackward) {
    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Squeeze(xs[0], Axes{0, 2, 4});
                return {y * y};  // to make it nonlinear
            },
            {(*testing::BuildArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<float>().WithPadding(1)).RequireGrad()},
            {(*testing::BuildArray({2, 3, 1, 4}).WithLinearData<float>(0.f, 0.1f)).RequireGrad()},
            {testing::BuildArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<float>()},
            {Full({1, 2, 1, 3, 1, 1, 4}, 1e-2f), Full({2, 3, 1, 4}, 1e-2f)},
            2,
            1e-4f,
            1e-3f);
}

TEST_THREAD_SAFE_P(ManipulationTest, BroadcastTo) {
    using T = int32_t;
    Shape input_shape{2, 3, 1};
    Shape output_shape{3, 1, 2, 3, 1, 2};

    Array aa = testing::BuildArray(input_shape).WithData<T>({1, 2, 3, 4, 5, 6});
    Array a = aa.At({Slice(), Slice(), Slice(), NewAxis{}});  // Make a broadcastable axis.
    ASSERT_EQ(Shape({2, 3, 1, 1}), a.shape());  // Check test precondition

    std::vector<T> output_data;
    for (int i = 0; i < 3; ++i) {
        output_data.insert(output_data.end(), {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
    }
    Array e = testing::BuildArray(output_shape).WithData<T>(output_data);

    Run([&]() {
        testing::CheckForward(
                [&output_shape](const std::vector<Array>& xs) {
                    Array y = BroadcastTo(xs[0], output_shape);
                    EXPECT_EQ(output_shape, y.shape());
                    EXPECT_EQ(xs[0].data().get(), y.data().get()) << "BroadcastTo must be done without copying data";
                    EXPECT_EQ(0, y.strides()[1]) << "Stride of broadcasted dimension must be 0";
                    return std::vector<Array>{y};
                },
                {a},
                {e});
    });
}

// Can't broadcast to smaller dimensions
TEST_P(ManipulationTest, InvalidBroadcastTo_NotEnoughDimension) {
    using T = int32_t;
    Shape input_shape{2, 3, 4};
    Shape output_shape{3, 4};

    Array a = testing::BuildArray(input_shape).WithLinearData<T>();
    EXPECT_THROW(BroadcastTo(a, output_shape), DimensionError);
}

// Can't broadcast with incompatible axis
TEST_P(ManipulationTest, InvalidBroadcastTo_IncompatibleDimension) {
    using T = int32_t;
    Shape input_shape{2, 3, 3};
    Shape output_shape{2, 4, 3};

    Array a = testing::BuildArray(input_shape).WithLinearData<T>();
    EXPECT_THROW(BroadcastTo(a, output_shape), DimensionError);
}

// Can't broadcast at the end
TEST_P(ManipulationTest, InvalidBroadcastTo_NotBroadcastableAtEnd) {
    using T = int32_t;
    Shape input_shape{2, 3};
    Shape output_shape{2, 3, 4};

    Array a = testing::BuildArray(input_shape).WithLinearData<T>();
    EXPECT_THROW(BroadcastTo(a, output_shape), DimensionError);
}

TEST_P(ManipulationTest, BroadcastToBackward) {
    using T = double;

    CheckBackward(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                return {BroadcastTo(xs[0], {2, 3, 4, 3})};
            },
            {(*testing::BuildArray({1, 3, 1, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>(-0.1, 0.1)},
            {Full({1, 3, 1, 3}, 1e-1, Dtype::kFloat64)});
}

TEST_P(ManipulationTest, BroadcastToDoubleBackward) {
    using T = double;

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = BroadcastTo(xs[0], {2, 3, 4, 3});
                return {y * y};  // to make it nonlinear
            },
            {(*testing::BuildArray({1, 3, 1, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {(*testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>(-0.1, 0.1)).RequireGrad()},
            {testing::BuildArray({1, 3, 1, 3}).WithLinearData<T>()},
            {Full({1, 3, 1, 3}, 1e-1, Dtype::kFloat64), Full({2, 3, 4, 3}, 1e-1, Dtype::kFloat64)});
}

TEST_THREAD_SAFE_P(ManipulationTest, Concatenate) {
    using T = int32_t;
    Shape input_shape{2, 3, 1};
    Shape output_shape{4, 3, 1};

    Array a = testing::BuildArray(input_shape).WithData<T>({1, 2, 3, 4, 5, 6});
    Array b = testing::BuildArray(input_shape).WithData<T>({7, 8, 9, 10, 11, 12});
    Array e = testing::BuildArray(output_shape).WithData<T>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    Array y = Concatenate(xs);
                    return std::vector<Array>{y};
                },
                {a, b},
                {e});
    });
}

TEST_P(ManipulationTest, ConcatenateAxis0) {
    using T = int32_t;
    Shape input_shape{2, 3, 1};
    Shape output_shape{4, 3, 1};

    Array a = testing::BuildArray(input_shape).WithData<T>({1, 2, 3, 4, 5, 6});
    Array b = testing::BuildArray(input_shape).WithData<T>({7, 8, 9, 10, 11, 12});
    Array e = testing::BuildArray(output_shape).WithData<T>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    EXPECT_ARRAY_EQ(e, Concatenate({a, b}, 0));
}

TEST_P(ManipulationTest, ConcatenateAxis1) {
    using T = int32_t;
    Shape input_shape{2, 3, 1};
    Shape output_shape{2, 6, 1};

    Array a = testing::BuildArray(input_shape).WithData<T>({1, 2, 3, 4, 5, 6});
    Array b = testing::BuildArray(input_shape).WithData<T>({7, 8, 9, 10, 11, 12});
    Array e = testing::BuildArray(output_shape).WithData<T>({1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12});

    EXPECT_ARRAY_EQ(e, Concatenate({a, b}, 1));
}

TEST_P(ManipulationTest, ConcatenateAxis2) {
    using T = int32_t;
    Shape input_shape{2, 3, 1};
    Shape output_shape{2, 3, 2};

    Array a = testing::BuildArray(input_shape).WithData<T>({1, 2, 3, 4, 5, 6});
    Array b = testing::BuildArray(input_shape).WithData<T>({7, 8, 9, 10, 11, 12});
    Array e = testing::BuildArray(output_shape).WithData<T>({1, 7, 2, 8, 3, 9, 4, 10, 5, 11, 6, 12});

    EXPECT_ARRAY_EQ(e, Concatenate({a, b}, 2));
}

TEST_P(ManipulationTest, ConcatenateAxisNone) {
    using T = int32_t;
    Shape input_shape{2, 3, 1};
    Shape output_shape{12};

    Array a = testing::BuildArray(input_shape).WithData<T>({1, 2, 3, 4, 5, 6});
    Array b = testing::BuildArray(input_shape).WithData<T>({7, 8, 9, 10, 11, 12});
    Array e = testing::BuildArray(output_shape).WithData<T>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    EXPECT_ARRAY_EQ(e, Concatenate({a, b}, nonstd::nullopt));
}

TEST_P(ManipulationTest, ConcatenateEmptyInput) { EXPECT_THROW(Concatenate({}), DimensionError); }

TEST_P(ManipulationTest, ConcatenateDifferentNdims) {
    using T = int32_t;

    Array a = testing::BuildArray({2, 3, 1}).WithData<T>({1, 2, 3, 4, 5, 6});
    Array b = testing::BuildArray({2, 3, 1, 1}).WithData<T>({7, 8, 9, 10, 11, 12});

    EXPECT_THROW(Concatenate({a, b}), DimensionError);
}

TEST_P(ManipulationTest, ConcatenateDifferentDtypes) {
    Shape input_shape{2, 3, 1};
    Shape output_shape{12};

    Array a = testing::BuildArray(input_shape).WithData<int32_t>({1, 2, 3, 4, 5, 6});
    Array b = testing::BuildArray(input_shape).WithData<int64_t>({7, 8, 9, 10, 11, 12});
    Array e = testing::BuildArray(output_shape).WithData<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    EXPECT_ARRAY_EQ(e, Concatenate({a, b}, nonstd::nullopt));
}

TEST_P(ManipulationTest, ConcatenateDifferentDimensionOnlyForConcatenationAxis) {
    using T = int32_t;

    Array a = testing::BuildArray({2, 3, 1}).WithData<T>({1, 2, 3, 4, 5, 6});
    Array b = testing::BuildArray({2, 2, 1}).WithData<T>({7, 8, 9, 10});
    Array e = testing::BuildArray({2, 5, 1}).WithData<T>({1, 2, 3, 7, 8, 4, 5, 6, 9, 10});

    EXPECT_ARRAY_EQ(e, Concatenate({a, b}, 1));
}

TEST_P(ManipulationTest, ConcatenateDifferentDimensionExceptForConcatenationAxis) {
    using T = int32_t;

    Array a = testing::BuildArray({2, 3, 1}).WithData<T>({1, 2, 3, 4, 5, 6});
    Array b = testing::BuildArray({2, 3, 2}).WithData<T>({7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});

    EXPECT_THROW(Concatenate({a, b}, 1), DimensionError);
}

TEST_P(ManipulationTest, ConcatenateNonContiguous) {
    using T = int32_t;

    Array aa = testing::BuildArray({1, 3, 1}).WithData<T>({1, 2, 3});
    Array a = BroadcastTo(aa, {2, 3, 1});
    Array bb = testing::BuildArray({2, 3, 2}).WithData<T>({7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18});
    Array b = bb.At({Slice{}, Slice{}, Slice{nonstd::nullopt, nonstd::nullopt, 2}});
    Array e = testing::BuildArray({2, 6, 1}).WithData<T>({1, 2, 3, 7, 9, 11, 1, 2, 3, 13, 15, 17});

    EXPECT_ARRAY_EQ(e, Concatenate({a, b}, 1));
}

TEST_P(ManipulationTest, ConcatenateBackward) {
    using T = double;
    Shape shape_x1{2, 2, 1};
    Shape shape_x2{3, 2, 1};
    Shape shape_y{5, 2, 1};

    Array x1 = (*testing::BuildArray(shape_x1).WithData<T>({1, 2, 3, 4})).RequireGrad();
    Array x2 = (*testing::BuildArray(shape_x2).WithData<T>({5, 6, 7, 8, 9, 10})).RequireGrad();
    Array gy = testing::BuildArray(shape_y).WithData<T>({3, 1, 4, 1, 5, 9, 2, 6, 5, 3});

    CheckBackward(
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {Concatenate(xs)}; },
            {x1, x2},
            {gy},
            {Full(shape_x1, 1e-6, Dtype::kFloat64), Full(shape_x2, 1e-6, Dtype::kFloat64)});
}

TEST_P(ManipulationTest, ConcatenateDoubleBackward) {
    using T = double;
    Shape shape_x1{2, 2, 1};
    Shape shape_x2{3, 2, 1};
    Shape shape_y{5, 2, 1};

    Array x1 = (*testing::BuildArray(shape_x1).WithData<T>({1, 2, 3, 4})).RequireGrad();
    Array x2 = (*testing::BuildArray(shape_x2).WithData<T>({5, 6, 7, 8, 9, 10})).RequireGrad();
    Array gy = (*testing::BuildArray(shape_y).WithData<T>({3, 1, 4, 1, 5, 9, 2, 6, 5, 3})).RequireGrad();
    Array ggx1 = testing::BuildArray(shape_x1).WithLinearData<T>(-0.1, 0.1);
    Array ggx2 = testing::BuildArray(shape_x2).WithLinearData<T>(-0.1, 0.1);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                Array y = Concatenate(xs);
                return {y * y};  // to make it nonlinear
            },
            {x1, x2},
            {gy},
            {ggx1, ggx2},
            {Full(shape_x1, 1e-6, Dtype::kFloat64), Full(shape_x2, 1e-6, Dtype::kFloat64), Full(shape_y, 1e-6, Dtype::kFloat64)});
}

TEST_THREAD_SAFE_P(ManipulationTest, Stack) {
    using T = int32_t;
    Shape input_shape{2, 3};
    Shape output_shape{3, 2, 3};

    Array a = testing::BuildArray(input_shape).WithData<T>({1, 2, 3, 4, 5, 6});
    Array b = testing::BuildArray(input_shape).WithData<T>({7, 8, 9, 10, 11, 12});
    Array e = testing::BuildArray(output_shape).WithData<T>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 7, 8, 9, 10, 11, 12});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    Array y = Stack(xs);
                    return std::vector<Array>{y};
                },
                {a, b, b},
                {e});
    });
}

TEST_THREAD_SAFE_P(ManipulationTest, StackAxis1) {
    using T = int32_t;
    Shape input_shape{2, 3};
    Shape output_shape{2, 3, 3};

    Array a = testing::BuildArray(input_shape).WithData<T>({1, 2, 3, 4, 5, 6});
    Array b = testing::BuildArray(input_shape).WithData<T>({7, 8, 9, 10, 11, 12});
    Array e = testing::BuildArray(output_shape).WithData<T>({1, 2, 3, 7, 8, 9, 7, 8, 9, 4, 5, 6, 10, 11, 12, 10, 11, 12});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    Array y = Stack(xs, 1);
                    return std::vector<Array>{y};
                },
                {a, b, b},
                {e});
    });
}

TEST_THREAD_SAFE_P(ManipulationTest, StackAxis2) {
    using T = int32_t;
    Shape input_shape{2, 3};
    Shape output_shape{2, 3, 3};

    Array a = testing::BuildArray(input_shape).WithData<T>({1, 2, 3, 4, 5, 6});
    Array b = testing::BuildArray(input_shape).WithData<T>({7, 8, 9, 10, 11, 12});
    Array e = testing::BuildArray(output_shape).WithData<T>({1, 7, 7, 2, 8, 8, 3, 9, 9, 4, 10, 10, 5, 11, 11, 6, 12, 12});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    Array y = Stack(xs, 2);
                    return std::vector<Array>{y};
                },
                {a, b, b},
                {e});
    });
}

TEST_P(ManipulationTest, StackDifferentShape) {
    using T = int32_t;
    Array a = testing::BuildArray(Shape{2, 1, 3}).WithData<T>({1, 2, 3, 4, 5, 6});
    Array b = testing::BuildArray(Shape{2, 2, 3}).WithData<T>({7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6});
    EXPECT_THROW(Stack({a, b}), DimensionError);
}

TEST_THREAD_SAFE_P(ManipulationTest, StackNonContiguous) {
    using T = int32_t;
    Array a = (*testing::BuildArray(Shape{1, 2}).WithData<T>({1, 2})).BroadcastTo({2, 2});
    Array b = testing::BuildArray(Shape{2, 2}).WithData<T>({3, 4, 5, 6}).WithPadding(1);
    Array e = testing::BuildArray(Shape{3, 2, 2}).WithData<T>({1, 2, 1, 2, 3, 4, 5, 6, 3, 4, 5, 6});
    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    Array y = Stack(xs);
                    return std::vector<Array>{y};
                },
                {a, b, b},
                {e});
    });
}

TEST_P(ManipulationTest, StackBackward) {
    using T = double;
    Shape shape_x{3, 2, 1};
    Shape shape_y{2, 3, 2, 1};

    Array x1 = (*testing::BuildArray(shape_x).WithData<T>({1, 2, 3, 4, 5, 6})).RequireGrad();
    Array x2 = (*testing::BuildArray(shape_x).WithData<T>({7, 8, 9, 10, 11, 12})).RequireGrad();
    Array gy = testing::BuildArray(shape_y).WithData<T>({3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8});

    CheckBackward(
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {Stack(xs)}; },
            {x1, x2},
            {gy},
            {Full(shape_x, 1e-6, Dtype::kFloat64), Full(shape_x, 1e-6, Dtype::kFloat64)});
}

TEST_P(ManipulationTest, StackDoubleBackward) {
    using T = double;
    Shape shape_x{3, 2, 1};
    Shape shape_y{2, 3, 2, 1};

    Array x1 = (*testing::BuildArray(shape_x).WithData<T>({1, 2, 3, 4, 5, 6})).RequireGrad();
    Array x2 = (*testing::BuildArray(shape_x).WithData<T>({7, 8, 9, 10, 11, 12})).RequireGrad();
    Array gy = (*testing::BuildArray(shape_y).WithData<T>({3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8})).RequireGrad();

    Array ggx1 = testing::BuildArray(shape_x).WithLinearData<T>(-0.1, 0.1);
    Array ggx2 = testing::BuildArray(shape_x).WithLinearData<T>(-0.1, 0.1);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                Array y = Stack(xs);
                return {y * y};  // to make it nonlinear
            },
            {x1, x2},
            {gy},
            {ggx1, ggx2},
            {Full(shape_x, 1e-6, Dtype::kFloat64), Full(shape_x, 1e-6, Dtype::kFloat64), Full(shape_y, 1e-6, Dtype::kFloat64)});
}

TEST_P(ManipulationTest, StackDoubleBackwardNegativeAxis) {
    using T = double;
    Shape shape_x{3, 2, 1};
    Shape shape_y{3, 2, 1, 2};

    Array x1 = (*testing::BuildArray(shape_x).WithData<T>({1, 2, 3, 4, 5, 6})).RequireGrad();
    Array x2 = (*testing::BuildArray(shape_x).WithData<T>({7, 8, 9, 10, 11, 12})).RequireGrad();
    Array gy = (*testing::BuildArray(shape_y).WithData<T>({3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8})).RequireGrad();

    Array ggx1 = testing::BuildArray(shape_x).WithLinearData<T>(-0.1, 0.1);
    Array ggx2 = testing::BuildArray(shape_x).WithLinearData<T>(-0.1, 0.1);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                Array y = Stack(xs, -1);
                return {y * y};  // to make it nonlinear
            },
            {x1, x2},
            {gy},
            {ggx1, ggx2},
            {Full(shape_x, 1e-6, Dtype::kFloat64), Full(shape_x, 1e-6, Dtype::kFloat64), Full(shape_y, 1e-6, Dtype::kFloat64)});
}

TEST_THREAD_SAFE_P(ManipulationTest, SplitSections) {
    Array a = testing::BuildArray({2, 4}).WithLinearData<int32_t>();
    Array e1 = testing::BuildArray({2, 2}).WithData<int32_t>({0, 1, 4, 5});
    Array e2 = testing::BuildArray({2, 2}).WithData<int32_t>({2, 3, 6, 7});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return Split(xs[0], 2, 1); }, {a}, {e1, e2}); });
}

TEST_THREAD_SAFE_P(ManipulationTest, SplitIndices) {
    Array a = testing::BuildArray({2, 4}).WithLinearData<int32_t>();
    Array e1 = testing::BuildArray({2, 2}).WithData<int32_t>({0, 1, 4, 5});
    Array e2 = testing::BuildArray({2, 1}).WithData<int32_t>({2, 6});
    Array e3 = testing::BuildArray({2, 1}).WithData<int32_t>({3, 7});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return Split(xs[0], {2, 3}, 1); }, {a}, {e1, e2, e3}); });
}

TEST_THREAD_SAFE_P(ManipulationTest, SplitDefaultAxesSections) {
    Array a = testing::BuildArray({2, 4}).WithLinearData<int32_t>();
    Array e1 = testing::BuildArray({1, 4}).WithData<int32_t>({0, 1, 2, 3});
    Array e2 = testing::BuildArray({1, 4}).WithData<int32_t>({4, 5, 6, 7});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return Split(xs[0], 2); }, {a}, {e1, e2}); });
}

TEST_THREAD_SAFE_P(ManipulationTest, SplitDefaultAxesIndices) {
    Array a = testing::BuildArray({2, 4}).WithLinearData<int32_t>();
    Array e1 = testing::BuildArray({1, 4}).WithData<int32_t>({0, 1, 2, 3});
    Array e2 = testing::BuildArray({1, 4}).WithData<int32_t>({4, 5, 6, 7});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return Split(xs[0], std::vector<int64_t>{1}); }, {a}, {e1, e2});
    });
}

TEST_THREAD_SAFE_P(ManipulationTest, SplitNoncontiguousSections) {
    Array a = testing::BuildArray({2, 4}).WithLinearData<int32_t>().WithPadding(1);
    Array e1 = testing::BuildArray({2, 2}).WithData<int32_t>({0, 1, 4, 5});
    Array e2 = testing::BuildArray({2, 2}).WithData<int32_t>({2, 3, 6, 7});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return Split(xs[0], 2, 1); }, {a}, {e1, e2}); });
}

TEST_THREAD_SAFE_P(ManipulationTest, SplitNoncontiguousIndices) {
    Array a = testing::BuildArray({2, 4}).WithLinearData<int32_t>().WithPadding(1);
    Array e1 = testing::BuildArray({2, 2}).WithData<int32_t>({0, 1, 4, 5});
    Array e2 = testing::BuildArray({2, 2}).WithData<int32_t>({2, 3, 6, 7});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return Split(xs[0], std::vector<int64_t>{2}, 1); }, {a}, {e1, e2});
    });
}

TEST_P(ManipulationTest, SplitSectionsBackward) {
    using T = double;

    CheckBackward(
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {Split(xs[0], 2, 0)}; },
            {(*testing::BuildArray({2, 4}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {testing::BuildArray({1, 4}).WithLinearData<T>(-0.1, 0.1), testing::BuildArray({1, 4}).WithLinearData<T>(-0.1, 0.1)},
            {Full({2, 4}, 1e-1, Dtype::kFloat64)});
}

TEST_P(ManipulationTest, SplitIndicesBackward) {
    using T = double;

    CheckBackward(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                return {Split(xs[0], {1, 3}, 1)};
            },
            {(*testing::BuildArray({2, 4}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {testing::BuildArray({2, 1}).WithLinearData<T>(-0.1, 0.1),
             testing::BuildArray({2, 2}).WithLinearData<T>(-0.1, 0.1),
             testing::BuildArray({2, 1}).WithLinearData<T>(-0.1, 0.1)},
            {Full({2, 4}, 1e-1, Dtype::kFloat64)});
}

TEST_P(ManipulationTest, SplitBackwardSomeOutputGradsAreAbsent) {
    using T = double;

    Array x = (*testing::BuildArray({4, 2}).WithLinearData<T>().WithPadding(1)).RequireGrad();
    Array gx1 = testing::BuildArray({4, 2}).WithData<T>({10, 11, 12, 13, 0, 0, 0, 0});
    Array gy1 = testing::BuildArray({2, 2}).WithData<T>({10, 11, 12, 13});
    std::vector<Array> y = Split(x, 2);
    EXPECT_EQ(y.size(), size_t{2});
    y[0].SetGrad(gy1);
    Backward(y[0]);
    EXPECT_ARRAY_EQ(gx1, *x.GetGrad());
}

TEST_P(ManipulationTest, SplitSectionsDoubleBackward) {
    using T = double;

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                std::vector<Array> ys = Split(xs[0], 2, 0);
                std::vector<Array> ys_non_linear;  // to make them nonlinear
                std::transform(ys.begin(), ys.end(), std::back_inserter(ys_non_linear), [](const Array& y) { return y * y; });
                return ys_non_linear;
            },
            {(*testing::BuildArray({2, 4}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {(*testing::BuildArray({1, 4}).WithLinearData<T>(-0.1, 0.1)).RequireGrad(),
             (*testing::BuildArray({1, 4}).WithLinearData<T>(-0.1, 0.1)).RequireGrad()},
            {testing::BuildArray({2, 4}).WithLinearData<T>()},
            {Full({2, 4}, 1e-1, Dtype::kFloat64), Full({1, 4}, 1e-1, Dtype::kFloat64), Full({1, 4}, 1e-1, Dtype::kFloat64)});
}

TEST_P(ManipulationTest, SplitIndicesDoubleBackward) {
    using T = double;

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                std::vector<Array> ys = Split(xs[0], {1, 3}, 1);
                std::vector<Array> ys_non_linear;  // to make them nonlinear
                std::transform(ys.begin(), ys.end(), std::back_inserter(ys_non_linear), [](const Array& y) { return y * y; });
                return ys_non_linear;
            },
            {(*testing::BuildArray({2, 4}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {(*testing::BuildArray({2, 1}).WithLinearData<T>(-0.1, 0.1)).RequireGrad(),
             (*testing::BuildArray({2, 2}).WithLinearData<T>(-0.1, 0.1)).RequireGrad(),
             (*testing::BuildArray({2, 1}).WithLinearData<T>(-0.1, 0.1)).RequireGrad()},
            {testing::BuildArray({2, 4}).WithLinearData<T>()},
            {Full({2, 4}, 1e-1, Dtype::kFloat64),
             Full({2, 1}, 1e-1, Dtype::kFloat64),
             Full({2, 2}, 1e-1, Dtype::kFloat64),
             Full({2, 1}, 1e-1, Dtype::kFloat64)});
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        ManipulationTest,
        ::testing::Values(
#ifdef CHAINERX_ENABLE_CUDA
                std::string{"cuda"},
#endif  // CHAINERX_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace chainerx
