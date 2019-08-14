#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "chainerx/array.h"
#include "chainerx/array_body.h"
#include "chainerx/array_node.h"
#include "chainerx/indexable_array.h"
#include "chainerx/indexer.h"
#include "chainerx/native/data_type.h"
#include "chainerx/numeric.h"

namespace chainerx {
namespace testing {
namespace testing_internal {

::testing::AssertionResult IsEqual(const char* a_expr, const char* b_expr, const Array& a, const Array& b);

::testing::AssertionResult IsAllClose(
        const char* a_expr,
        const char* b_expr,
        const char* rtol_expr,
        const char* atol_expr,
        const char* equal_nan_expr,
        const Array& a,
        const Array& b,
        double rtol,
        double atol,
        bool equal_nan);

::testing::AssertionResult HaveDistinctArrayNodes(const char* a_expr, const char* b_expr, const Array& a, const Array& b);

}  // namespace testing_internal

namespace array_check_detail {

// Checks if `subset` is a subset of `superset` and `superset` have no duplicate elements.
template <typename SubsetContainer, typename SupersetContainer>
bool IsSubset(const SubsetContainer& subset, const SupersetContainer& superset) {
    return std::all_of(subset.begin(), subset.end(), [&superset](const auto& l) {
        return std::count_if(superset.begin(), superset.end(), [&l](const auto& r) { return l == r; }) == 1;
    });
}

template <typename ActualContainer, typename T = typename ActualContainer::value_type>
::testing::AssertionResult IsSetEqual(const std::vector<T>& expected, const ActualContainer& actual) {
    bool result = IsSubset(expected, actual) && IsSubset(actual, expected);
    if (result) {
        return ::testing::AssertionSuccess();
    }
    ::testing::AssertionResult os = ::testing::AssertionFailure();
    os << "Expected : { ";
    for (auto it = expected.begin(); it != expected.end(); ++it) {
        os << *it;
        if (it != std::prev(expected.end())) {
            os << ", ";
        }
    }
    os << " }\nTo be equal to { ";
    for (auto it = actual.begin(); it != actual.end(); ++it) {
        os << *it;
        if (it != std::prev(actual.end())) {
            os << ", ";
        }
    }
    os << " }\n";
    return os;
}

}  // namespace array_check_detail

inline ::testing::AssertionResult IsBackpropIdsEqual(const std::vector<BackpropId>& expected, const Array& array) {
    std::vector<BackpropId> actual;
    std::vector<std::shared_ptr<internal::ArrayNode>>& nodes = internal::GetArrayBody(array)->nodes();
    actual.reserve(nodes.size());
    std::transform(nodes.begin(), nodes.end(), std::back_inserter(actual), [](const std::shared_ptr<internal::ArrayNode>& node) {
        return node->backprop_id();
    });
    return array_check_detail::IsSetEqual(expected, actual);
}

// TODO(hvy): Allow friendlier failure messages by avoiding EXPECT_* and return ::testing::AssertionResult instead.
template <typename T, typename Container>
void ExpectDataEqual(Container&& expected_data, const Array& actual) {
    Array native_actual = actual.ToNative();
    IndexableArray<const T> actual_iarray{native_actual};
    Indexer<> indexer{actual.shape()};
    for (auto it = indexer.It(0); it; ++it) {
        T actual_value = actual_iarray[it];
        int64_t i = it.raw_index();
        EXPECT_EQ(expected_data[i], actual_value) << "where i is " << i;
    }
}

// TODO(hvy): Allow friendlier failure messages by avoiding EXPECT_* and return ::testing::AssertionResult instead.
template <typename T>
void ExpectDataEqual(T expected, const Array& actual) {
    Array native_actual = actual.ToNative();
    VisitDtype(actual.dtype(), [&](auto pt) {
        using U = typename decltype(pt)::type;
        IndexableArray<const U> actual_iarray{native_actual};
        Indexer<> indexer{actual.shape()};
        for (auto it = indexer.It(0); it; ++it) {
            U actual_value = native::StorageToDataType<const U>(actual_iarray[it]);
            if (IsNan(expected)) {
                EXPECT_TRUE(IsNan(actual_value)) << "where i is " << it.raw_index();
            } else {
                EXPECT_EQ(expected, actual_value) << "where i is " << it.raw_index();
            }
        }
    });
}

}  // namespace testing

// Expects that given arrays have same elements and that they belong to the same device.
#define EXPECT_ARRAY_EQ(a, b) EXPECT_PRED_FORMAT2(testing::testing_internal::IsEqual, a, b)

// Expects that the given arrays have elements that are all close to each other and that they belong to the same device.
//
// GET_MACRO is used to "overload" EXPECT_ARRAY_ALL_CLOSE with optional arguments.
#define GET_MACRO(_1, _2, _3, _4, _5, NAME, ...) NAME
#define EXPECT_ARRAY_ALL_CLOSE2(a, b) EXPECT_PRED_FORMAT5(testing::testing_internal::IsAllClose, a, b, 1e-5, 1e-8, false)
#define EXPECT_ARRAY_ALL_CLOSE3(a, b, rtol) EXPECT_PRED_FORMAT5(testing::testing_internal::IsAllClose, a, b, rtol, 1e-8, false)
#define EXPECT_ARRAY_ALL_CLOSE4(a, b, rtol, atol) EXPECT_PRED_FORMAT5(testing::testing_internal::IsAllClose, a, b, rtol, atol, false)
#define EXPECT_ARRAY_ALL_CLOSE5(a, b, rtol, atol, equal_nan) \
    EXPECT_PRED_FORMAT5(testing::testing_internal::IsAllClose, a, b, rtol, atol, equal_nan)
#define EXPECT_ARRAY_ALL_CLOSE(...)                                                                                            \
    GET_MACRO(__VA_ARGS__, EXPECT_ARRAY_ALL_CLOSE5, EXPECT_ARRAY_ALL_CLOSE4, EXPECT_ARRAY_ALL_CLOSE3, EXPECT_ARRAY_ALL_CLOSE2) \
    (__VA_ARGS__)

// Expects that the arrays a and b have distint array bodies.
#define EXPECT_ARRAY_HAVE_DISTINCT_ARRAY_NODES(a, b) EXPECT_PRED_FORMAT2(testing::testing_internal::HaveDistinctArrayNodes, a, b)

}  // namespace chainerx
