#pragma once

#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>

#include "xchainer/array.h"
#include "xchainer/array_body.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"

namespace xchainer {
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
    IndexableArray<const T> actual_iarray{native_actual};
    Indexer<> indexer{actual.shape()};
    for (auto it = indexer.It(0); it; ++it) {
        T actual_value = actual_iarray[it];
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual_value)) << "where i is " << it.raw_index();
        } else {
            EXPECT_EQ(expected, actual_value) << "where i is " << it.raw_index();
        }
    }
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

}  // namespace xchainer
