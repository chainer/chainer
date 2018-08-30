#include "xchainer/axes.h"

#include <cstdint>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "xchainer/error.h"

namespace xchainer {
namespace {

void CheckSpanEqual(std::initializer_list<int8_t> expect, gsl::span<const int8_t> actual) {
    EXPECT_EQ(gsl::make_span(expect.begin(), expect.end()), actual);
}

TEST(AxesTest, Ctor) {
    {  // Default ctor
        const Axes axes{};
        EXPECT_EQ(0, axes.ndim());
        EXPECT_EQ(size_t{0}, axes.size());
    }
    {  // From std::initializer_list
        const Axes axes{2, 0, 3};
        EXPECT_EQ(3, axes.ndim());
        EXPECT_EQ(size_t{3}, axes.size());
        CheckSpanEqual({2, 0, 3}, axes.span());
    }
    {  // From gsl::span
        const std::array<int8_t, 3> dims{2, 0, 3};
        const Axes axes{gsl::make_span(dims)};
        EXPECT_EQ(3, axes.ndim());
        CheckSpanEqual({2, 0, 3}, axes.span());
    }
    {  // From iterators
        const std::vector<int8_t> dims{2, 0, 3};
        const Axes axes{dims.begin(), dims.end()};
        EXPECT_EQ(3, axes.ndim());
        CheckSpanEqual({2, 0, 3}, axes.span());
    }
    {  // From empty std::initializer_list
        const Axes axes(std::initializer_list<int8_t>{});
        EXPECT_EQ(0, axes.ndim());
        CheckSpanEqual({}, axes.span());
    }
    {  // From empty gsl::span
        const std::array<int8_t, 0> dims{};
        const Axes axes{gsl::make_span(dims)};
        EXPECT_EQ(0, axes.ndim());
        CheckSpanEqual({}, axes.span());
    }
    {  // From empty iterators
        const std::vector<int8_t> dims{};
        const Axes axes{dims.begin(), dims.end()};
        EXPECT_EQ(0, axes.ndim());
        CheckSpanEqual({}, axes.span());
    }
    {  // Too long std::initializer_list
        EXPECT_THROW(Axes({0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1}), DimensionError);
    }
    {  // Too long gsl::span
        const std::array<int8_t, kMaxNdim + 1> too_long{1};
        EXPECT_THROW(Axes{gsl::make_span(too_long)}, DimensionError);
    }
    {  // Too long iterators
        std::vector<int8_t> dims{};
        dims.reserve(kMaxNdim + 1);
        for (int i = 0; i < kMaxNdim + 1; ++i) {
            dims.emplace_back(i % 3);
        }
        EXPECT_THROW(Axes({dims.begin(), dims.end()}), DimensionError);
    }
}

TEST(AxesTest, Subscript) {
    const Axes axes = {2, 0, 3};
    EXPECT_EQ(2, axes[0]);
    EXPECT_EQ(0, axes[1]);
    EXPECT_EQ(3, axes[2]);
    EXPECT_THROW(axes[-1], DimensionError);
    EXPECT_THROW(axes[3], DimensionError);
}

TEST(AxesTest, Compare) {
    {
        const Axes axes = {2, 0, 3};
        const Axes axes2 = {2, 0, 3};
        EXPECT_TRUE(axes == axes2);
    }
    {
        const Axes axes = {2, 0, 3};
        const Axes axes2 = {2, 3};
        EXPECT_TRUE(axes != axes2);
    }
    {
        const Axes axes = {2, 0, 3};
        const Axes axes2 = {4, 2, 1};
        EXPECT_TRUE(axes != axes2);
    }
}

TEST(AxesTest, Iterator) {
    const Axes axes = {2, 0, 3};
    CheckSpanEqual({2, 0, 3}, gsl::make_span(axes.begin(), axes.end()));
    CheckSpanEqual({3, 0, 2}, gsl::make_span(std::vector<int8_t>{axes.rbegin(), axes.rend()}));
}

TEST(AxesTest, ToString) {
    {
        const Axes axes = {};
        EXPECT_EQ(axes.ToString(), "()");
    }
    {
        const Axes axes = {4};
        EXPECT_EQ(axes.ToString(), "(4,)");
    }
    {
        const Axes axes = {2, 0, 3};
        EXPECT_EQ(axes.ToString(), "(2, 0, 3)");
    }
}

TEST(AxesTest, SpanFromAxes) {
    const Axes axes = {2, 3, 4};
    CheckSpanEqual({2, 3, 4}, gsl::make_span(axes));
}

using GetSortedAxesNormalTestParam = std::tuple<Axes, int8_t, Axes>;

class GetSortedAxesNormalTest : public ::testing::TestWithParam<GetSortedAxesNormalTestParam> {};

TEST_P(GetSortedAxesNormalTest, Check) {
    const Axes& axis = std::get<0>(GetParam());
    int8_t ndim = std::get<1>(GetParam());
    const Axes& expect = std::get<2>(GetParam());
    EXPECT_EQ(expect, internal::GetSortedAxes(axis, ndim));
    EXPECT_EQ(expect, internal::GetSortedAxesOrAll(axis, ndim));
}

INSTANTIATE_TEST_CASE_P(
        ForEachInputs,
        GetSortedAxesNormalTest,
        ::testing::Values(
                GetSortedAxesNormalTestParam{Axes{}, 0, Axes{}},
                GetSortedAxesNormalTestParam{{1}, 2, {1}},
                GetSortedAxesNormalTestParam{{-1}, 2, {1}},
                GetSortedAxesNormalTestParam{{1, 0}, 2, {0, 1}},
                GetSortedAxesNormalTestParam{{2, -2}, 3, {1, 2}}));

using GetSortedAxesInvalidTestParam = std::tuple<Axes, int8_t>;

class GetSortedAxesInvalidTest : public ::testing::TestWithParam<GetSortedAxesInvalidTestParam> {};

TEST_P(GetSortedAxesInvalidTest, Invalid) {
    const Axes& axis = std::get<0>(GetParam());
    int8_t ndim = std::get<1>(GetParam());
    EXPECT_THROW(internal::GetSortedAxes(axis, ndim), DimensionError);
    EXPECT_THROW(internal::GetSortedAxesOrAll(axis, ndim), DimensionError);
}

INSTANTIATE_TEST_CASE_P(
        ForEachInputs,
        GetSortedAxesInvalidTest,
        ::testing::Values(
                GetSortedAxesInvalidTestParam{{0}, 0},
                GetSortedAxesInvalidTestParam{{2}, 1},
                GetSortedAxesInvalidTestParam{{-2}, 1},
                GetSortedAxesInvalidTestParam{{0, 2, 1}, 2},
                GetSortedAxesInvalidTestParam{{0, 0}, 1}));

using GetSortedAxesOrAllTestParam = std::tuple<int8_t, Axes>;

class GetSortedAxesOrAllTest : public ::testing::TestWithParam<GetSortedAxesOrAllTestParam> {};

TEST_P(GetSortedAxesOrAllTest, All) {
    int8_t axis = std::get<0>(GetParam());
    const Axes& expect = std::get<1>(GetParam());
    EXPECT_EQ(expect, internal::GetSortedAxesOrAll(nonstd::nullopt, axis));
}

INSTANTIATE_TEST_CASE_P(
        ForEachNdim,
        GetSortedAxesOrAllTest,
        ::testing::Values(
                GetSortedAxesOrAllTestParam{0, {}},
                GetSortedAxesOrAllTestParam{1, {0}},
                GetSortedAxesOrAllTestParam{2, {0, 1}},
                GetSortedAxesOrAllTestParam{3, {0, 1, 2}},
                GetSortedAxesOrAllTestParam{4, {0, 1, 2, 3}}));

}  // namespace
}  // namespace xchainer
