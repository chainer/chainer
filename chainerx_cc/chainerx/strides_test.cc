#include "chainerx/strides.h"

#include <numeric>
#include <vector>

#include <absl/types/span.h>
#include <gtest/gtest.h>

#include "chainerx/axes.h"

namespace chainerx {
namespace {

void CheckSpanEqual(std::initializer_list<int64_t> expect, absl::Span<const int64_t> actual) {
    EXPECT_EQ(absl::MakeConstSpan(expect.begin(), expect.end()), actual);
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
    {  // From span
        const std::array<int64_t, 3> dims{48, 16, 4};
        const Strides strides{absl::MakeConstSpan(dims)};
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
    {  // From empty span
        const std::array<int64_t, 0> dims{};
        const Strides strides{absl::MakeConstSpan(dims)};
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
    {  // Too long span
        const std::array<int64_t, kMaxNdim + 1> too_long{1};
        EXPECT_THROW(Strides{absl::MakeConstSpan(too_long)}, DimensionError);
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
    CheckSpanEqual({48, 16, 4}, absl::MakeConstSpan(std::vector<int64_t>{strides.begin(), strides.end()}));
    CheckSpanEqual({4, 16, 48}, absl::MakeConstSpan(std::vector<int64_t>{strides.rbegin(), strides.rend()}));
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
    CheckSpanEqual({2, 3, 4}, absl::MakeConstSpan(strides));
}

TEST(StridesTest, Permute) {
    const Strides strides = {2, 3, 4};
    CheckSpanEqual({3, 4}, strides.Permute(Axes{1, 2}).span());
    EXPECT_THROW(strides.Permute(Axes{3}), DimensionError);
    EXPECT_THROW(strides.Permute(Axes{-1}), DimensionError);
}

struct GetDataRangeTestParams {
    Shape shape;
    Strides strides;
    size_t itemsize;
    int64_t first;
    int64_t last;
};

class GetDataRangeTest : public ::testing::TestWithParam<GetDataRangeTestParams> {};
INSTANTIATE_TEST_CASE_P(
        GetDataRangeTest,
        GetDataRangeTest,
        ::testing::Values(
                GetDataRangeTestParams{{2, 3, 4}, {96, 32, 8}, 8, 0, 192},
                GetDataRangeTestParams{{10, 12}, {160, 8}, 8, 0, 1536},
                GetDataRangeTestParams{{}, {}, 8, 0, 8},
                GetDataRangeTestParams{{3, 0, 3}, {24, 24, 8}, 8, 0, 0},
                GetDataRangeTestParams{{10, 3, 4}, {-96, 32, 8}, 8, -864, 96},
                GetDataRangeTestParams{{10, 3, 4}, {-96, -32, -8}, 8, -952, 8},
                GetDataRangeTestParams{{3, 4}, {8, 24}, 8, 0, 96},
                GetDataRangeTestParams{{100}, {24}, 8, 0, 2384}));

TEST_P(GetDataRangeTest, GetDataRange) {
    GetDataRangeTestParams param = GetParam();
    std::tuple<int64_t, int64_t> actual = GetDataRange(param.shape, param.strides, param.itemsize);
    EXPECT_EQ(actual, std::make_tuple(param.first, param.last));
}

}  // namespace
}  // namespace chainerx
