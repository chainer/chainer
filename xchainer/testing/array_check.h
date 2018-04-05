#pragma once

#include <cmath>
#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "xchainer/array.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/numeric.h"

// TODO(hvy): Make it independent from gtest.
namespace xchainer {
namespace testing {

inline void ExpectAllClose(const Array& expected, const Array& actual, double rtol, double atol, bool equal_nan = false) {
    EXPECT_EQ(&expected.device(), &actual.device());
    EXPECT_TRUE(AllClose(expected, actual, rtol, atol, equal_nan));
}

inline void ExpectEqual(const Array& expected, const Array& actual) { ExpectAllClose(expected, actual, 0., 0., true); }

inline void ExpectEqualCopy(const Array& expected, const Array& actual) {
    EXPECT_TRUE(actual.IsContiguous());
    EXPECT_EQ(0, actual.offset());

    // Deep copy, therefore assert different addresses to data
    EXPECT_NE(expected.data().get(), actual.data().get());

    ExpectEqual(expected, actual);
}

inline void ExpectEqualView(const Array& expected, const Array& actual) {
    EXPECT_EQ(expected.dtype(), actual.dtype());
    EXPECT_EQ(expected.shape(), actual.shape());
    EXPECT_EQ(expected.IsContiguous(), actual.IsContiguous());
    EXPECT_EQ(expected.offset(), actual.offset());

    // Shallow copy, therefore assert the same address to data
    EXPECT_EQ(expected.data().get(), actual.data().get());

    // Views should have different array bodies.
    EXPECT_NE(expected.body(), actual.body());

    ExpectEqual(expected, actual);
}

template <typename T>
void ExpectDataEqual(const T* expected_data, const Array& actual) {
    Array native_actual = actual.ToNative();
    IndexableArray<const T> actual_iarray{native_actual};
    Indexer indexer{actual.shape()};
    for (int64_t i = 0; i < indexer.total_size(); ++i) {
        indexer.Set(i);
        T actual_value = actual_iarray[indexer];
        EXPECT_EQ(expected_data[i], actual_value) << "where i is " << i;
    }
}

template <typename T>
void ExpectDataEqual(T expected, const Array& actual) {
    Array native_actual = actual.ToNative();
    IndexableArray<const T> actual_iarray{native_actual};
    Indexer indexer{actual.shape()};
    for (int64_t i = 0; i < indexer.total_size(); ++i) {
        indexer.Set(i);
        T actual_value = actual_iarray[indexer];
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual_value)) << "where i is " << i;
        } else {
            EXPECT_EQ(expected, actual_value) << "where i is " << i;
        }
    }
}

}  // namespace testing
}  // namespace xchainer
