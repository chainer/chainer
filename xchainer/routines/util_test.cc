#include "xchainer/routines/util.h"

#include <cstdint>
#include <tuple>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/error.h"
#include "xchainer/ndim_vector.h"

namespace xchainer {
namespace internal {
namespace {

using GetSortedAxesNormalTestParam = std::tuple<NdimVector<int8_t>, int8_t, NdimVector<int8_t>>;

class GetSortedAxesNormalTest : public ::testing::TestWithParam<GetSortedAxesNormalTestParam> {};

TEST_P(GetSortedAxesNormalTest, Check) {
    const auto& axis = std::get<0>(GetParam());
    int8_t ndim = std::get<1>(GetParam());
    const auto& expect = std::get<2>(GetParam());
    EXPECT_EQ(expect, GetSortedAxes(axis, ndim));
    EXPECT_EQ(expect, GetSortedAxesOrAll(axis, ndim));
}

INSTANTIATE_TEST_CASE_P(
        ForEachInputs,
        GetSortedAxesNormalTest,
        ::testing::Values(
                GetSortedAxesNormalTestParam{{}, 0, {}},
                GetSortedAxesNormalTestParam{{1}, 2, {1}},
                GetSortedAxesNormalTestParam{{-1}, 2, {1}},
                GetSortedAxesNormalTestParam{{1, 0}, 2, {0, 1}},
                GetSortedAxesNormalTestParam{{2, -2}, 3, {1, 2}}));

using GetSortedAxesInvalidTestParam = std::tuple<NdimVector<int8_t>, int8_t>;

class GetSortedAxesInvalidTest : public ::testing::TestWithParam<GetSortedAxesInvalidTestParam> {};

TEST_P(GetSortedAxesInvalidTest, Invalid) {
    const auto& axis = std::get<0>(GetParam());
    int8_t ndim = std::get<1>(GetParam());
    EXPECT_THROW(GetSortedAxes(axis, ndim), DimensionError);
    EXPECT_THROW(GetSortedAxesOrAll(axis, ndim), DimensionError);
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

using GetSortedAxesOrAllTestParam = std::tuple<int8_t, NdimVector<int8_t>>;

class GetSortedAxesOrAllTest : public ::testing::TestWithParam<GetSortedAxesOrAllTestParam> {};

TEST_P(GetSortedAxesOrAllTest, All) {
    int8_t axis = std::get<0>(GetParam());
    const auto& expect = std::get<1>(GetParam());
    EXPECT_EQ(expect, GetSortedAxesOrAll(nonstd::nullopt, axis));
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
}  // namespace internal
}  // namespace xchainer
