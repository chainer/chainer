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

::testing::AssertionResult IsCopy(const char* orig_expr, const char* copy_expr, const Array& orig, const Array& copy);

::testing::AssertionResult IsView(const char* orig_expr, const char* view_expr, const Array& orig, const Array& view);

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
#define XCHAINER_EXPECT_EQ(a, b) EXPECT_PRED_FORMAT2(testing::testing_internal::IsEqual, a, b)

// Expects that the given arrays have elements that are all close to each other and that they belong to the same device.
//
// GET_MACRO is used to "overload" XCHAINER_EXPECT_ARRAY_ALL_CLOSE with optional arguments.
#define GET_MACRO(_1, _2, _3, _4, _5, NAME, ...) NAME
#define XCHAINER_EXPECT_ARRAY_ALL_CLOSE2(a, b) EXPECT_PRED_FORMAT5(testing::testing_internal::IsAllClose, a, b, 1e-5, 1e-8, false)
#define XCHAINER_EXPECT_ARRAY_ALL_CLOSE3(a, b, rtol) EXPECT_PRED_FORMAT5(testing::testing_internal::IsAllClose, a, b, rtol, 1e-8, false)
#define XCHAINER_EXPECT_ARRAY_ALL_CLOSE4(a, b, rtol, atol) \
    EXPECT_PRED_FORMAT5(testing::testing_internal::IsAllClose, a, b, rtol, atol, false)
#define XCHAINER_EXPECT_ARRAY_ALL_CLOSE5(a, b, rtol, atol, equal_nan) \
    EXPECT_PRED_FORMAT5(testing::testing_internal::IsAllClose, a, b, rtol, atol, equal_nan)
#define XCHAINER_EXPECT_ARRAY_ALL_CLOSE(...)  \
    GET_MACRO(                                \
            __VA_ARGS__,                      \
            XCHAINER_EXPECT_ARRAY_ALL_CLOSE5, \
            XCHAINER_EXPECT_ARRAY_ALL_CLOSE4, \
            XCHAINER_EXPECT_ARRAY_ALL_CLOSE3, \
            XCHAINER_EXPECT_ARRAY_ALL_CLOSE2) \
    (__VA_ARGS__)

// Expects that the given array b is a valid copy of a.
#define XCHAINER_EXPECT_ARRAY_COPY_EQ(orig, copy) EXPECT_PRED_FORMAT2(testing::testing_internal::IsCopy, orig, copy)

// Expects that the given array b is a valid view of a.
#define XCHAINER_EXPECT_ARRAY_VIEW_EQ(orig, view) EXPECT_PRED_FORMAT2(testing::testing_internal::IsView, orig, view)

}  // namespace xchainer
