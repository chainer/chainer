#pragma once

#include <cmath>
#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "xchainer/array.h"
#include "xchainer/backend.h"
#include "xchainer/context.h"
#include "xchainer/device.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/native/native_backend.h"

// TODO(hvy): Make it independent from gtest.
namespace xchainer {
namespace testing {

namespace {

Array ToNativeDevice(const Array& array) {
    Context& context = array.device().backend().context();
    Backend& native_backend = context.GetBackend(native::NativeBackend::kDefaultName);
    Device& native_device = native_backend.GetDevice(0);
    Array native_array = array.ToDevice(native_device);

    native_device.Synchronize();
    return native_array;
}

}  // namespace

template <typename T>
void ExpectDataEqual(const Array& expected, const Array& actual) {
    actual.device().Synchronize();
    Array native_expected = ToNativeDevice(expected);
    Array native_actual = ToNativeDevice(actual);
    IndexableArray<const T> expected_iarray{native_expected};
    IndexableArray<const T> actual_iarray{native_actual};
    Indexer<> indexer{actual.shape()};
    for (int64_t i = 0; i < indexer.total_size(); ++i) {
        indexer.Set(i);
        T expected_value = expected_iarray[indexer];
        T actual_value = actual_iarray[indexer];
        if (std::isnan(expected_value)) {
            EXPECT_TRUE(std::isnan(actual_value)) << "where i is " << i;
        } else {
            EXPECT_EQ(expected_value, actual_value) << "where i is " << i;
        }
    }
}

template <typename T>
void ExpectDataEqual(const T* expected_data, const Array& actual) {
    actual.device().Synchronize();
    Array native_actual = ToNativeDevice(actual);
    IndexableArray<const T> actual_iarray{native_actual};
    Indexer<> indexer{actual.shape()};
    for (int64_t i = 0; i < indexer.total_size(); ++i) {
        indexer.Set(i);
        T actual_value = actual_iarray[indexer];
        EXPECT_EQ(expected_data[i], actual_value) << "where i is " << i;
    }
}

template <typename T>
void ExpectDataEqual(T expected, const Array& actual) {
    actual.device().Synchronize();
    Array native_actual = ToNativeDevice(actual);
    IndexableArray<const T> actual_iarray{native_actual};
    Indexer<> indexer{actual.shape()};
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

inline void ExpectArraysEqualAttributes(const Array& a, const Array& b) {
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
