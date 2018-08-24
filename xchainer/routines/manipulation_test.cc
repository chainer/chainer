#include "xchainer/routines/manipulation.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/array_index.h"
#include "xchainer/axes.h"
#include "xchainer/check_backward.h"
#include "xchainer/device_id.h"
#include "xchainer/error.h"
#include "xchainer/shape.h"
#include "xchainer/strides.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/array_check.h"
#include "xchainer/testing/device_session.h"
#include "xchainer/testing/routines.h"
#include "xchainer/testing/threading.h"

namespace xchainer {
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

TEST_P(ManipulationTest, AsScalar) {
    using T = float;
    T value = 2.0f;
    Array a = testing::BuildArray({1, 1, 1}).WithData<T>({value}).WithPadding(1);

    testing::RunThreads(
            [&a, &value]() {
                using T = float;
                Scalar s = AsScalar(a);
                EXPECT_EQ(s.dtype(), TypeToDtype<T>);
                EXPECT_EQ(static_cast<T>(s), value);
            },
            // TODO(sonots): Run concurrency test in CUDA
            GetParam() == "cuda" ? 0 : 2);
}

TEST_P(ManipulationTest, AsScalarInvalidZeroElement) {
    Array a = testing::BuildArray({0}).WithData<float>({});
    EXPECT_THROW(AsScalar(a), DimensionError);
}

TEST_P(ManipulationTest, AsScalarInvalidMoreThanOneElements) {
    Array a = testing::BuildArray({2}).WithData<float>({1.0f, 2.0f});
    EXPECT_THROW(AsScalar(a), DimensionError);
}

TEST_P(ManipulationTest, RollAxis) {
    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<int32_t>();
    Array e = testing::BuildArray({3, 2, 4}).WithData<int32_t>(
            {0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23});

    testing::CheckForward(
            [](const std::vector<Array>& xs) { return std::vector<Array>{RollAxis(xs[0], 1)}; },
            {a},
            {e},
            // TODO(sonots): Run concurrency test in CUDA
            GetParam() == "cuda" ? 0 : 1);
}

TEST_P(ManipulationTest, RollAxisWithStart) {
    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<int32_t>();
    Array e = testing::BuildArray({3, 2, 4}).WithData<int32_t>(
            {0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23});

    testing::CheckForward(
            [](const std::vector<Array>& xs) { return std::vector<Array>{RollAxis(xs[0], -3, -1)}; },
            {a},
            {e},
            // TODO(sonots): Run concurrency test in CUDA
            GetParam() == "cuda" ? 0 : 1);
}

TEST_P(ManipulationTest, Transpose) {
    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<int32_t>();
    Array e = testing::BuildArray({4, 2, 3}).WithData<int32_t>(
            {0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23});

    testing::CheckForward(
            [](const std::vector<Array>& xs) {
                Array y = Transpose(xs[0], {2, 0, 1});
                EXPECT_EQ(Strides({4, 48, 16}), y.strides());
                return std::vector<Array>{y};
            },
            {a},
            {e},
            // TODO(sonots): Run concurrency test in CUDA
            GetParam() == "cuda" ? 0 : 1);
}

TEST_P(ManipulationTest, TransposeDefaultAxes) {
    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<int32_t>();
    Array e = testing::BuildArray({4, 3, 2}).WithData<int32_t>(
            {0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23});

    testing::CheckForward(
            [](const std::vector<Array>& xs) {
                Array y = Transpose(xs[0]);
                EXPECT_EQ(Strides({4, 16, 48}), y.strides());
                return std::vector<Array>{y};
            },
            {a},
            {e},
            // TODO(sonots): Run concurrency test in CUDA
            GetParam() == "cuda" ? 0 : 1);
}

TEST_P(ManipulationTest, TransposeNoncontiguous) {
    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<int32_t>().WithPadding(1);
    Array e = testing::BuildArray({4, 2, 3}).WithData<int32_t>(
            {0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23});

    testing::CheckForward(
            [](const std::vector<Array>& xs) {
                return std::vector<Array>{Transpose(xs[0], {2, 0, 1})};
            },
            {a},
            {e},
            // TODO(sonots): Run concurrency test in CUDA
            GetParam() == "cuda" ? 0 : 1);
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

TEST_P(ManipulationTest, Reshape) {
    using T = int32_t;
    Shape input_shape{2, 3, 4};
    Shape output_shape{3, 4, 2};

    Array a = testing::BuildArray(input_shape).WithLinearData<T>();
    Array e = testing::BuildArray(output_shape).WithLinearData<T>();

    testing::CheckForward(
            [&output_shape](const std::vector<Array>& xs) {
                Array y = Reshape(xs[0], output_shape);
                EXPECT_EQ(xs[0].data().get(), y.data().get()) << "Reshape must be done without copying data";
                return std::vector<Array>{y};
            },
            {a},
            {e},
            // TODO(sonots): Run concurrency test in CUDA
            GetParam() == "cuda" ? 0 : 1);
}

// #461
TEST_P(ManipulationTest, ReshapeWithStrideOne) {
    using T = bool;
    Shape input_shape{6};
    Shape output_shape{2, 3};

    Array a = testing::BuildArray(input_shape).WithLinearData<T>();
    Array e = testing::BuildArray(output_shape).WithLinearData<T>();

    testing::CheckForward(
            [&output_shape](const std::vector<Array>& xs) {
                Array y = Reshape(xs[0], output_shape);
                EXPECT_EQ(xs[0].data().get(), y.data().get()) << "Reshape must be done without copying data";
                return std::vector<Array>{y};
            },
            {a},
            {e},
            // TODO(sonots): Run concurrency test in CUDA
            GetParam() == "cuda" ? 0 : 1);
}

// #461
TEST_P(ManipulationTest, ReshapeNewAxisAtEnd) {
    using T = double;
    Shape input_shape{2, 4};
    Shape output_shape{2, 1, 4, 1};

    Array a = testing::BuildArray(input_shape).WithLinearData<T>();
    Array e = testing::BuildArray(output_shape).WithLinearData<T>();

    testing::CheckForward(
            [&output_shape](const std::vector<Array>& xs) {
                Array y = Reshape(xs[0], output_shape);
                EXPECT_EQ(xs[0].data().get(), y.data().get()) << "Reshape must be done without copying data";
                return std::vector<Array>{y};
            },
            {a},
            {e},
            // TODO(sonots): Run concurrency test in CUDA
            GetParam() == "cuda" ? 0 : 1);
}

// If an input array has a unit-length axis with 0-stride, that axis should not give rise to any copies.
TEST_P(ManipulationTest, ReshapeNoCopyZeroStrideAxis) {
    using T = int32_t;
    Shape input_shape_before_newaxis{2, 3, 4};
    Shape output_shape{3, 4, 2};

    // The shape of the input array is (2, 1, 3, 4) with strides (48, 0, 16, 4).
    Array a = (*testing::BuildArray(input_shape_before_newaxis).WithLinearData<T>()).At({Slice{}, NewAxis{}, Slice{}, Slice{}});
    ASSERT_TRUE(std::find(a.strides().begin(), a.strides().end(), 0) != a.strides().end());
    Array e = testing::BuildArray(output_shape).WithLinearData<T>();

    testing::CheckForward(
            [&output_shape](const std::vector<Array>& xs) {
                Array y = Reshape(xs[0], output_shape);
                EXPECT_EQ(xs[0].data().get(), y.data().get()) << "Reshape must be done without copying data";
                return std::vector<Array>{y};
            },
            {a},
            {e},
            // TODO(sonots): Run concurrency test in CUDA
            GetParam() == "cuda" ? 0 : 1);
}

TEST_P(ManipulationTest, ReshapeWithCopy) {
    using T = int32_t;
    Shape input_shape{2, 3, 4};
    Shape output_shape{2, 12};

    Array a = testing::BuildArray(input_shape).WithLinearData<T>().WithPadding(1);
    Array e = testing::BuildArray(output_shape).WithLinearData<T>();

    testing::CheckForward(
            [&output_shape](const std::vector<Array>& xs) {
                Array y = Reshape(xs[0], output_shape);
                EXPECT_NE(xs[0].data().get(), y.data().get()) << "Reshape must be done with copy";
                return std::vector<Array>{y};
            },
            {a},
            {e},
            // TODO(sonots): Run concurrency test in CUDA
            GetParam() == "cuda" ? 0 : 1);
}

TEST_P(ManipulationTest, InvalidReshape) {
    using T = int32_t;
    Shape input_shape{2, 3, 4};
    Shape output_shape{2, 4, 4};

    Array a = testing::BuildArray(input_shape).WithLinearData<T>();
    EXPECT_THROW(Reshape(a, output_shape), DimensionError);
}

TEST_P(ManipulationTest, SqueezeAllUnitLengthAxes) {
    using T = int32_t;

    Array a = testing::BuildArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<T>();
    Array e = testing::BuildArray({2, 3, 4}).WithLinearData<T>();

    testing::CheckForward(
            [](const std::vector<Array>& xs) { return std::vector<Array>{Squeeze(xs[0])}; },
            {a},
            {e},
            // TODO(sonots): Run concurrency test in CUDA
            GetParam() == "cuda" ? 0 : 1);
}

TEST_P(ManipulationTest, SqueezeSpecifiedUnitLenghtAxes) {
    using T = int32_t;

    Array a = testing::BuildArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<T>();
    Array e = testing::BuildArray({2, 3, 1, 4}).WithLinearData<T>();

    testing::CheckForward(
            [](const std::vector<Array>& xs) {
                return std::vector<Array>{Squeeze(xs[0], Axes{2, 0, 4})};
            },
            {a},
            {e},
            // TODO(sonots): Run concurrency test in CUDA
            GetParam() == "cuda" ? 0 : 1);
}

TEST_P(ManipulationTest, SqueezeAllAxes) {
    using T = int32_t;

    Array a = testing::BuildArray({1, 1, 1}).WithLinearData<T>();
    Array e = testing::BuildArray({}).WithData<T>({0});

    testing::CheckForward(
            [](const std::vector<Array>& xs) { return std::vector<Array>{Squeeze(xs[0])}; },
            {a},
            {e},
            // TODO(sonots): Run concurrency test in CUDA
            GetParam() == "cuda" ? 0 : 1);
}

TEST_P(ManipulationTest, SqueezeMultipleCalls) {
    using T = int32_t;

    Array a = testing::BuildArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<T>();
    Array e = testing::BuildArray({2, 3, 1, 4}).WithLinearData<T>();

    testing::CheckForward(
            [](const std::vector<Array>& xs) {
                return std::vector<Array>{Squeeze(Squeeze(xs[0], Axes{0, 2}), Axes{3})};
            },
            {a},
            {e},
            // TODO(sonots): Run concurrency test in CUDA
            GetParam() == "cuda" ? 0 : 1);
}

TEST_P(ManipulationTest, SqueezeNonContiguous) {
    using T = int32_t;

    Array a = testing::BuildArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<T>().WithPadding(1);
    Array e = testing::BuildArray({2, 3, 1, 4}).WithLinearData<T>();

    testing::CheckForward(
            [](const std::vector<Array>& xs) {
                return std::vector<Array>{Squeeze(xs[0], Axes{0, 2, 4})};
            },
            {a},
            {e},
            // TODO(sonots): Run concurrency test in CUDA
            GetParam() == "cuda" ? 0 : 1);
}

TEST_P(ManipulationTest, SqueezeNegativeAxis) {
    using T = int32_t;

    Array a = testing::BuildArray({2, 3, 4, 1}).WithLinearData<T>();
    Array e = testing::BuildArray({2, 3, 4}).WithLinearData<T>();

    testing::CheckForward(
            [](const std::vector<Array>& xs) { return std::vector<Array>{Squeeze(xs[0], Axes{-1})}; },
            {a},
            {e},
            // TODO(sonots): Run concurrency test in CUDA
            GetParam() == "cuda" ? 0 : 1);
}

TEST_P(ManipulationTest, SqueezeNoSqueezableAxes) {
    using T = int32_t;

    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<T>();

    testing::CheckForward(
            [](const std::vector<Array>& xs) {
                Array y = Squeeze(xs[0]);
                EXPECT_EQ(internal::GetArrayBody(y), internal::GetArrayBody(xs[0]));
                return std::vector<Array>{y};
            },
            {a},
            {a},
            // TODO(sonots): Run concurrency test in CUDA
            GetParam() == "cuda" ? 0 : 1);
}

TEST_P(ManipulationTest, InvalidSqueezeNonUnitLengthAxis) {
    using T = int32_t;

    Array a = testing::BuildArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<T>();
    EXPECT_THROW(Array b = Squeeze(a, Axes{1}), DimensionError);
}

TEST_P(ManipulationTest, InvalidSqueezeDuplicateAxes) {
    using T = int32_t;

    Array a = testing::BuildArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<T>();
    EXPECT_THROW(Squeeze(a, Axes{0, 2, 2}), XchainerError);
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
            1e-4f,
            1e-3f);
}

TEST_P(ManipulationTest, BroadcastTo) {
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

    testing::CheckForward(
            [&output_shape](const std::vector<Array>& xs) {
                Array y = BroadcastTo(xs[0], output_shape);
                EXPECT_EQ(output_shape, y.shape());
                EXPECT_EQ(xs[0].data().get(), y.data().get()) << "BroadcastTo must be done without copying data";
                EXPECT_EQ(0, y.strides()[1]) << "Stride of broadcasted dimension must be 0";
                return std::vector<Array>{y};
            },
            {a},
            {e},
            // TODO(sonots): Run concurrency test in CUDA
            GetParam() == "cuda" ? 0 : 1);
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
            {Full({1, 3, 1, 3}, 1e-1)});
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
            {Full({1, 3, 1, 3}, 1e-1), Full({2, 3, 4, 3}, 1e-1)});
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        ManipulationTest,
        ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace xchainer
