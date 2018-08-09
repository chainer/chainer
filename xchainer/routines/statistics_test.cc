#include "xchainer/routines/statistics.h"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/check_backward.h"
#include "xchainer/device_id.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/routines/creation.h"
#include "xchainer/scalar.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/array_check.h"
#include "xchainer/testing/device_session.h"

namespace xchainer {
namespace {

class StatisticsTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        const std::string& backend_name = GetParam();
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

private:
    nonstd::optional<testing::DeviceSession> device_session_;
};

TEST_P(StatisticsTest, Mean) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1);
    Array b = Mean(a, Axes{2, 1, -1});
    EXPECT_EQ(Shape{2}, b.shape());
    Array e = testing::BuildArray({2}).WithData<T>({17.5f, 53.5f});
    testing::ExpectEqual(e, b);
}

TEST_P(StatisticsTest, MeanAllAxes) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 3}).WithLinearData<T>().WithPadding(1);
    Array b = Mean(a);
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray({}).WithData<T>({8.5f});
    testing::ExpectEqual(e, b);
}

TEST_P(StatisticsTest, MeanZero) {
    using T = float;

    Array a = testing::BuildArray({0}).WithData<T>({});
    Array b = Mean(a);
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray({}).WithData<T>({std::nanf("")});
    testing::ExpectEqual(e, b);
}

TEST_P(StatisticsTest, MeanOne) {
    using T = float;

    Array a = testing::BuildArray({}).WithData<T>({42.0f}).WithPadding(1);
    Array b = Mean(a);
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray({}).WithData<T>({42.0f});
    testing::ExpectEqual(e, b);
}

TEST_P(StatisticsTest, MeanTwo) {
    using T = float;

    Array a = testing::BuildArray({2}).WithData<T>({42.0f, 37.0f}).WithPadding(1);
    Array b = Mean(a);
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray({}).WithData<T>({39.5f});
    testing::ExpectEqual(e, b);
}

TEST_P(StatisticsTest, MeanLarge) {
    using T = double;

    Array a = testing::BuildArray({0x100000}).WithLinearData<T>().WithPadding(1);
    Array b = Mean(a, Axes{0});
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray({}).WithData<T>({524287.5f});
    testing::ExpectEqual(e, b);
}

TEST_P(StatisticsTest, MeanKeepDims) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 2, 4}).WithLinearData<T>().WithPadding(1);
    Array b = Mean(a, Axes{-1, 1}, true);
    EXPECT_EQ(Shape({2, 1, 2, 1}), b.shape());
    EXPECT_EQ(0, b.strides()[1]);
    EXPECT_EQ(0, b.strides()[3]);
    Array e = testing::BuildArray({2, 1, 2, 1}).WithData<T>({9.5f, 13.5f, 33.5f, 37.5f});
    testing::ExpectEqual(e, b);
}

TEST_P(StatisticsTest, InvalidMeanDuplicateAxes) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<T>();
    EXPECT_THROW(Mean(a, Axes{1, 1}), XchainerError);
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
            {Full({2, 3, 4, 3}, 1e-3)});
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
            {Full({2, 3, 4, 3}, 1e-3), Full({2, 1, 4, 1}, 1e-3)});
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
            {Full({2, 3, 4, 3}, 1e-3), Full({2, 4}, 1e-3)});
}

TEST_P(StatisticsTest, Var) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1);
    Array b = Var(a, Axes{2, 1, -1});
    EXPECT_EQ(Shape{2}, b.shape());
    Array e = testing::BuildArray({2}).WithData<T>({107.91666667f, 107.91666667f});
    testing::ExpectAllClose(e, b);
}

TEST_P(StatisticsTest, VarAllAxes) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 3}).WithLinearData<T>().WithPadding(1);
    Array b = Var(a);
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray({}).WithData<T>({26.91666667f});
    testing::ExpectAllClose(e, b);
}

TEST_P(StatisticsTest, VarZero) {
    using T = float;

    Array a = testing::BuildArray({0}).WithData<T>({});
    Array b = Var(a);
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray({}).WithData<T>({std::nanf("")});
    testing::ExpectEqual(e, b);
}

TEST_P(StatisticsTest, VarOne) {
    using T = float;

    Array a = testing::BuildArray({}).WithData<T>({42.0f}).WithPadding(1);
    Array b = Var(a);
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray({}).WithData<T>({0.f});
    testing::ExpectEqual(e, b);
}

TEST_P(StatisticsTest, VarTwo) {
    using T = float;

    Array a = testing::BuildArray({2}).WithData<T>({42.0f, 37.0f}).WithPadding(1);
    Array b = Var(a);
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray({}).WithData<T>({6.25f});
    testing::ExpectEqual(e, b);
}

TEST_P(StatisticsTest, VarLarge) {
    using T = double;

    Array a = testing::BuildArray({0x100000}).WithLinearData<T>().WithPadding(1);
    Array b = Var(a, Axes{0});
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::BuildArray({}).WithData<T>({91625968981.25});
    testing::ExpectAllClose(e, b);
}

TEST_P(StatisticsTest, VarKeepDims) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 2, 4}).WithLinearData<T>().WithPadding(1);
    Array b = Var(a, Axes{-1, 1}, true);
    EXPECT_EQ(Shape({2, 1, 2, 1}), b.shape());
    EXPECT_EQ(0, b.strides()[1]);
    EXPECT_EQ(0, b.strides()[3]);
    Array e = testing::BuildArray({2, 1, 2, 1}).WithData<T>({43.91666667f, 43.91666667f, 43.91666667f, 43.91666667f});
    testing::ExpectEqual(e, b);
}

TEST_P(StatisticsTest, InvalidVarDuplicateAxes) {
    using T = float;

    Array a = testing::BuildArray({2, 3, 4}).WithLinearData<T>();
    EXPECT_THROW(Var(a, Axes{1, 1}), XchainerError);
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
            {Full({2, 3, 4, 3}, 1e-3)});
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
            {Full({2, 3, 4, 3}, 1e-3), Full({2, 1, 4, 1}, 1e-3)});
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
            {Full({2, 3, 4, 3}, 1e-3), Full({2, 4}, 1e-3)});
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        StatisticsTest,
        ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace xchainer
