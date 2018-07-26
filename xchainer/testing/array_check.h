#pragma once

#include <cmath>
#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "xchainer/array.h"
#include "xchainer/array_body.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/numeric.h"

// TODO(hvy): Make it independent from gtest.
namespace xchainer {
namespace testing {

inline void ExpectAllClose(const Array& expected, const Array& actual, double rtol = 1e-5, double atol = 1e-8, bool equal_nan = false) {
    EXPECT_EQ(&expected.device(), &actual.device());
    EXPECT_TRUE(AllClose(expected, actual, rtol, atol, equal_nan)) << "Expected: " << expected << "\nActual: " << actual;
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
    EXPECT_NE(internal::GetArrayBody(expected), internal::GetArrayBody(actual));

    // No array nodes should be shared.
    for (const std::shared_ptr<internal::ArrayNode>& array_node_e : internal::GetArrayBody(expected)->nodes()) {
        for (const std::shared_ptr<internal::ArrayNode>& array_node_a : internal::GetArrayBody(actual)->nodes()) {
            EXPECT_NE(array_node_e, array_node_a);
        }
    }

    ExpectEqual(expected, actual);
}

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
}  // namespace xchainer
