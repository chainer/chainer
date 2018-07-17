#include "xchainer/strides.h"

#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include <gsl/gsl>

namespace xchainer {
namespace {

void CheckSpanEqual(std::initializer_list<int64_t> expect, gsl::span<const int64_t> actual) {
    EXPECT_EQ(gsl::make_span(expect.begin(), expect.end()), actual);
}

TEST(StridesTest, Ctor) {
    {  // Default ctor
        const Strides strides{};
        EXPECT_EQ(0, strides.ndim());
        EXPECT_EQ(size_t{0}, strides.size());
    }
    {  // From std::initializer_list
        const Strides strides{48, 16, 4};
        EXPECT_EQ(3, strides.ndim());
        EXPECT_EQ(size_t{3}, strides.size());
        CheckSpanEqual({48, 16, 4}, strides.span());
    }
    {  // From gsl::span
        const std::array<int64_t, 3> dims{48, 16, 4};
        const Strides strides{gsl::make_span(dims)};
        EXPECT_EQ(3, strides.ndim());
        CheckSpanEqual({48, 16, 4}, strides.span());
    }
    {  // From iterators
        const std::vector<int64_t> dims{48, 16, 4};
        const Strides strides{dims.begin(), dims.end()};
        EXPECT_EQ(3, strides.ndim());
        CheckSpanEqual({48, 16, 4}, strides.span());
    }
    {  // From empty std::initializer_list
        const Strides strides(std::initializer_list<int64_t>{});
        EXPECT_EQ(0, strides.ndim());
        CheckSpanEqual({}, strides.span());
    }
    {  // From empty gsl::span
        const std::array<int64_t, 0> dims{};
        const Strides strides{gsl::make_span(dims)};
        EXPECT_EQ(0, strides.ndim());
        CheckSpanEqual({}, strides.span());
    }
    {  // From empty iterators
        const std::vector<int64_t> dims{};
        const Strides strides{dims.begin(), dims.end()};
        EXPECT_EQ(0, strides.ndim());
        CheckSpanEqual({}, strides.span());
    }
    {  // From shape and element size
        const Strides strides{{2, 3, 4}, 4};
        EXPECT_EQ(3, strides.ndim());
        EXPECT_EQ(size_t{3}, strides.size());
        CheckSpanEqual({48, 16, 4}, strides.span());
    }
    {  // From shape and dtype
        const Strides strides{{2, 3, 4}, Dtype::kInt32};
        EXPECT_EQ(3, strides.ndim());
        EXPECT_EQ(size_t{3}, strides.size());
        CheckSpanEqual({48, 16, 4}, strides.span());
    }
    {  // Too long std::initializer_list
        EXPECT_THROW(Strides({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}), DimensionError);
    }
    {  // Too long gsl::span
        const std::array<int64_t, kMaxNdim + 1> too_long{1};
        EXPECT_THROW(Strides{gsl::make_span(too_long)}, DimensionError);
    }
    {  // Too long iterators
        std::vector<int64_t> dims{};
        dims.resize(kMaxNdim + 1);
        std::iota(dims.begin(), dims.end(), int64_t{1});
        EXPECT_THROW(Strides({dims.begin(), dims.end()}), DimensionError);
    }
}

TEST(StridesTest, Subscript) {
    const Strides strides = {48, 16, 4};
    EXPECT_EQ(48, strides[0]);
    EXPECT_EQ(16, strides[1]);
    EXPECT_EQ(4, strides[2]);
    EXPECT_THROW(strides[-1], DimensionError);
    EXPECT_THROW(strides[3], DimensionError);
}

TEST(StridesTest, Compare) {
    {
        const Strides strides = {48, 16, 4};
        const Strides strides2 = {48, 16, 4};
        EXPECT_TRUE(strides == strides2);
    }
    {
        const Strides strides = {48, 16, 4};
        const Strides strides2 = {48, 16};
        EXPECT_TRUE(strides != strides2);
    }
    {
        const Strides strides = {48, 16, 4};
        const Strides strides2 = {4, 8, 24};
        EXPECT_TRUE(strides != strides2);
    }
}

TEST(StridesTest, CheckEqual) {
    {
        const Strides strides = {48, 16, 4};
        const Strides strides2 = {48, 16, 4};
        CheckEqual(strides, strides2);
    }
    {
        const Strides strides = {48, 16, 4};
        const Strides strides2 = {};
        EXPECT_THROW(CheckEqual(strides, strides2), DimensionError);
    }
}

TEST(StridesTest, Iterator) {
    const Strides strides = {48, 16, 4};
    CheckSpanEqual({48, 16, 4}, gsl::make_span(strides.begin(), strides.end()));
    CheckSpanEqual({4, 16, 48}, gsl::make_span(std::vector<int64_t>{strides.rbegin(), strides.rend()}));
}

TEST(StridesTest, ToString) {
    {
        const Strides strides = {};
        EXPECT_EQ(strides.ToString(), "()");
    }
    {
        const Strides strides = {4};
        EXPECT_EQ(strides.ToString(), "(4,)");
    }
    {
        const Strides strides = {48, 16, 4};
        EXPECT_EQ(strides.ToString(), "(48, 16, 4)");
    }
}

TEST(StridesTest, SpanFromStrides) {
    const Strides strides = {2, 3, 4};
    CheckSpanEqual({2, 3, 4}, gsl::make_span(strides));
}

using GetDataRangeTestType = std::tuple<Shape, Strides, size_t, int64_t, int64_t>;
class GetDataRangeTest : public ::testing::TestWithParam<GetDataRangeTestType> {};
INSTANTIATE_TEST_CASE_P(
        GetDataRangeTest,
        GetDataRangeTest,
        ::testing::Values(
                GetDataRangeTestType{{2, 3, 4}, {96, 32, 8}, 8, 0, 192},
                GetDataRangeTestType{{10, 12}, {160, 8}, 8, 0, 1536},
                GetDataRangeTestType{{}, {}, 8, 0, 8},
                GetDataRangeTestType{{3, 0, 3}, {24, 24, 8}, 8, 0, 0},
                GetDataRangeTestType{{10, 3, 4}, {-96, 32, 8}, 8, -864, 96},
                GetDataRangeTestType{{10, 3, 4}, {-96, -32, -8}, 8, -952, 8},
                GetDataRangeTestType{{3, 4}, {8, 24}, 8, 0, 96},
                GetDataRangeTestType{{100}, {24}, 8, 0, 2384}));

TEST_P(GetDataRangeTest, GetDataRange) {
    Shape shape;
    Strides strides;
    size_t itemsize;
    int64_t expected_first, expected_last, actual_first, actual_last;
    std::tie(shape, strides, itemsize, expected_first, expected_last) = GetParam();
    std::tie(actual_first, actual_last) = GetDataRange(shape, strides, itemsize);
    EXPECT_EQ(actual_first, expected_first);
    EXPECT_EQ(actual_last, expected_last);
}

}  // namespace
}  // namespace xchainer
