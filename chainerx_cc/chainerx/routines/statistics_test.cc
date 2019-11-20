// TODO(niboshi): Write Python test and delete this file.

#include "chainerx/routines/statistics.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <absl/types/optional.h>
#include <gtest/gtest.h>

#include "chainerx/array.h"
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

class StatisticsTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        const std::string& backend_name = GetParam();
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

private:
    absl::optional<testing::DeviceSession> device_session_;
};

TEST_P(StatisticsTest, Mean) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1);
    Array b = Mean(a, Axes{2, 1, -1});
    EXPECT_EQ(Shape{2}, b.shape());
    Array e = testing::BuildArray({2}).WithData<T>({17.5f, 53.5f});
    EXPECT_ARRAY_EQ(e, b);
}

TEST_THREAD_SAFE_P(StatisticsTest, MeanAllAxes) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 3}).WithLinearData<T>().WithPadding(1);
    Array e = testing::BuildArray({}).WithData<T>({8.5f});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Mean(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(StatisticsTest, MeanZero) {
    using T = float;

    Array a = testing::BuildArray({0}).WithData<T>({});
    Array e = testing::BuildArray({}).WithData<T>({std::nanf("")});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Mean(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(StatisticsTest, MeanOne) {
    using T = float;

    Array a = testing::BuildArray({}).WithData<T>({42.0f}).WithPadding(1);
    Array e = testing::BuildArray({}).WithData<T>({42.0f});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Mean(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(StatisticsTest, MeanTwo) {
    using T = float;

    Array a = testing::BuildArray({2}).WithData<T>({42.0f, 37.0f}).WithPadding(1);
    Array e = testing::BuildArray({}).WithData<T>({39.5f});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Mean(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(StatisticsTest, MeanLarge) {
    using T = double;

    Array a = testing::BuildArray({0x100000}).WithLinearData<T>().WithPadding(1);
    Array e = testing::BuildArray({}).WithData<T>({524287.5f});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Mean(xs[0], Axes{0})}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(StatisticsTest, MeanKeepDims) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 2, 4}).WithLinearData<T>().WithPadding(1);
    Array e = testing::BuildArray({2, 1, 2, 1}).WithData<T>({9.5f, 13.5f, 33.5f, 37.5f});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    Array y = Mean(xs[0], Axes{-1, 1}, true);
                    EXPECT_EQ(0, y.strides()[1]);
                    EXPECT_EQ(0, y.strides()[3]);
                    return std::vector<Array>{y};
                },
                {a},
                {e});
    });
}

TEST_P(StatisticsTest, InvalidMeanDuplicateAxes) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<T>();
    EXPECT_THROW(Mean(a, Axes{1, 1}), ChainerxError);
}

TEST_P(StatisticsTest, InvalidMeanOutOfRangeAxes) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<T>();
    EXPECT_THROW(Mean(a, Axes{3}), DimensionError);
}

TEST_P(StatisticsTest, MeanBackward) {
    using T = double;

    CheckBackward(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                return {Mean(xs[0], Axes{1, 3})};
            },
            {(*testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {testing::BuildArray({2, 4}).WithLinearData<T>(-0.1, 0.1)},
            {Full({2, 3, 4, 3}, 1e-3, Dtype::kFloat64)});
}

TEST_P(StatisticsTest, MeanDoubleBackward_Keepdims) {
    using T = double;

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Mean(xs[0], Axes{1, 3}, true);
                return {y * y};  // to make it nonlinear
            },
            {(*testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {(*testing::BuildArray({2, 1, 4, 1}).WithLinearData<T>(-0.1, 0.1)).RequireGrad()},
            {testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>()},
            {Full({2, 3, 4, 3}, 1e-3, Dtype::kFloat64), Full({2, 1, 4, 1}, 1e-3, Dtype::kFloat64)});
}

TEST_P(StatisticsTest, MeanDoubleBackward_NoKeepdims) {
    using T = double;

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Mean(xs[0], Axes{1, 3}, false);
                return {y * y};  // to make it nonlinear
            },
            {(*testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {(*testing::BuildArray({2, 4}).WithLinearData<T>(-0.1, 0.1)).RequireGrad()},
            {testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>()},
            {Full({2, 3, 4, 3}, 1e-3, Dtype::kFloat64), Full({2, 4}, 1e-3, Dtype::kFloat64)});
}

TEST_THREAD_SAFE_P(StatisticsTest, Var) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1);
    Array e = testing::BuildArray({2}).WithData<T>({107.91666667f, 107.91666667f});

    Run([&]() {
        testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Var(xs[0], Axes{2, 1, -1})}; }, {a}, {e});
    });
}

TEST_THREAD_SAFE_P(StatisticsTest, VarAllAxes) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 3}).WithLinearData<T>().WithPadding(1);
    Array e = testing::BuildArray({}).WithData<T>({26.91666667f});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Var(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(StatisticsTest, VarZero) {
    using T = float;

    Array a = testing::BuildArray({0}).WithData<T>({});
    Array e = testing::BuildArray({}).WithData<T>({std::nanf("")});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Var(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(StatisticsTest, VarOne) {
    using T = float;

    Array a = testing::BuildArray({}).WithData<T>({42.0f}).WithPadding(1);
    Array e = testing::BuildArray({}).WithData<T>({0.f});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Var(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(StatisticsTest, VarTwo) {
    using T = float;

    Array a = testing::BuildArray({2}).WithData<T>({42.0f, 37.0f}).WithPadding(1);
    Array e = testing::BuildArray({}).WithData<T>({6.25f});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Var(xs[0])}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(StatisticsTest, VarLarge) {
    using T = double;

    Array a = testing::BuildArray({0x100000}).WithLinearData<T>().WithPadding(1);
    Array e = testing::BuildArray({}).WithData<T>({91625968981.25});

    Run([&]() { testing::CheckForward([](const std::vector<Array>& xs) { return std::vector<Array>{Var(xs[0], Axes{0})}; }, {a}, {e}); });
}

TEST_THREAD_SAFE_P(StatisticsTest, VarKeepDims) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 2, 4}).WithLinearData<T>().WithPadding(1);
    Array e = testing::BuildArray({2, 1, 2, 1}).WithData<T>({43.91666667f, 43.91666667f, 43.91666667f, 43.91666667f});

    Run([&]() {
        testing::CheckForward(
                [](const std::vector<Array>& xs) {
                    Array y = Var(xs[0], Axes{-1, 1}, true);
                    EXPECT_EQ(0, y.strides()[1]);
                    EXPECT_EQ(0, y.strides()[3]);
                    return std::vector<Array>{y};
                },
                {a},
                {e});
    });
}

TEST_P(StatisticsTest, InvalidVarDuplicateAxes) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<T>();
    EXPECT_THROW(Var(a, Axes{1, 1}), ChainerxError);
}

TEST_P(StatisticsTest, InvalidVarOutOfRangeAxes) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<T>();
    EXPECT_THROW(Var(a, Axes{3}), DimensionError);
}

TEST_P(StatisticsTest, VarBackward) {
    using T = double;

    CheckBackward(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                return {Var(xs[0], Axes{1, 3})};
            },
            {(*testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {testing::BuildArray({2, 4}).WithLinearData<T>(-0.1, 0.1)},
            {Full({2, 3, 4, 3}, 1e-3, Dtype::kFloat64)});
}

TEST_P(StatisticsTest, VarDoubleBackward_Keepdims) {
    using T = double;

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                return {Var(xs[0], Axes{1, 3}, true)};
            },
            {(*testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {(*testing::BuildArray({2, 1, 4, 1}).WithLinearData<T>(-0.1, 0.1)).RequireGrad()},
            {testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>()},
            {Full({2, 3, 4, 3}, 1e-3, Dtype::kFloat64), Full({2, 1, 4, 1}, 1e-3, Dtype::kFloat64)});
}

TEST_P(StatisticsTest, VarDoubleBackward_NoKeepdims) {
    using T = double;

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                return {Var(xs[0], Axes{1, 3}, false)};
            },
            {(*testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {(*testing::BuildArray({2, 4}).WithLinearData<T>(-0.1, 0.1)).RequireGrad()},
            {testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>()},
            {Full({2, 3, 4, 3}, 1e-3, Dtype::kFloat64), Full({2, 4}, 1e-3, Dtype::kFloat64)});
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        StatisticsTest,
        ::testing::Values(
#ifdef CHAINERX_ENABLE_CUDA
                std::string{"cuda"},
#endif  // CHAINERX_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace chainerx
