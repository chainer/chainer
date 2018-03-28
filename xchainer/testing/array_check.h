#include <cmath>
#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "xchainer/array.h"
#include "xchainer/device.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"

// TODO(hvy): Make it independent from gtest.
namespace xchainer {
namespace testing {

template <typename T>
void ExpectDataEqual(const Array& expected, const Array& actual) {
    actual.device().Synchronize();
    IndexableArray<const T> expected_iarray{expected};
    IndexableArray<const T> actual_iarray{actual};
    Indexer<> indexer{actual.shape()};
    for (int64_t i = 0; i < indexer.total_size(); ++i) {
        indexer.Set(i);
        const auto& expected = expected_iarray[indexer];
        const auto& actual = actual_iarray[indexer];
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual)) << "where i is " << i;
        } else {
            EXPECT_EQ(expected, actual) << "where i is " << i;
        }
    }
}

template <typename T>
void ExpectDataEqual(const T* expected_data, const Array& actual) {
    actual.device().Synchronize();
    IndexableArray<const T> actual_iarray{actual};
    Indexer<> indexer{actual.shape()};
    for (int64_t i = 0; i < indexer.total_size(); ++i) {
        indexer.Set(i);
        const auto& actual = actual_iarray[indexer];
        EXPECT_EQ(expected_data[i], actual) << "where i is " << i;
    }
}

template <typename T>
void ExpectDataEqual(T expected, const Array& actual) {
    actual.device().Synchronize();
    IndexableArray<const T> actual_iarray{actual};
    Indexer<> indexer{actual.shape()};
    for (int64_t i = 0; i < indexer.total_size(); ++i) {
        indexer.Set(i);
        const auto& actual = actual_iarray[indexer];
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual)) << "where i is " << i;
        } else {
            EXPECT_EQ(expected, actual) << "where i is " << i;
        }
    }
}

template <typename T>
void ExpectEqual(const Array& expected, const Array& actual) {
    EXPECT_EQ(expected.dtype(), actual.dtype());
    EXPECT_EQ(expected.shape(), actual.shape());
    EXPECT_EQ(&expected.device(), &actual.device());
    ExpectDataEqual<T>(expected, actual);
}

template <typename T>
void ExpectEqualCopy(const Array& expected, const Array& actual) {
    EXPECT_EQ(expected.dtype(), actual.dtype());
    EXPECT_EQ(expected.shape(), actual.shape());
    EXPECT_EQ(&expected.device(), &actual.device());

    // Deep copy, therefore assert different addresses to data
    EXPECT_NE(expected.data().get(), actual.data().get());

    EXPECT_TRUE(actual.IsContiguous());
    EXPECT_EQ(0, actual.offset());

    ExpectDataEqual<T>(expected, actual);
}

void ExpectArraysEqualAttributes(const Array& a, const Array& b) {
    EXPECT_EQ(a.dtype(), b.dtype());
    EXPECT_EQ(a.shape(), b.shape());
    EXPECT_EQ(a.IsContiguous(), b.IsContiguous());
    EXPECT_EQ(a.offset(), b.offset());
}

template <typename T>
void ExpectEqualView(const Array& expected, const Array& actual) {
    ExpectEqual<T>(expected, actual);
    ExpectArraysEqualAttributes(expected, actual);

    // Shallow copy, therefore assert the same address to data
    EXPECT_EQ(expected.data().get(), actual.data().get());
    EXPECT_EQ(&expected.device(), &actual.device());

    // Views should have different array bodies.
    EXPECT_NE(expected.body(), actual.body());
}

}  // namespace testing
}  // namespace xchainer
