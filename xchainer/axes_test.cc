#include "xchainer/axes.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include <gsl/gsl>

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
        EXPECT_THROW(Axes({0, 1, 2, 0, 1, 2, 0, 1, 2}), DimensionError);
    }
    {  // Too long gsl::span
        const std::array<int8_t, kMaxNdim + 1> too_long{1};
        EXPECT_THROW(Axes{gsl::make_span(too_long)}, DimensionError);
    }
    {  // Too long iterators
        const std::vector<int8_t> dims{0, 1, 2, 0, 1, 2, 0, 1, 2};
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

}  // namespace
}  // namespace xchainer
