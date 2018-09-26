#include "chainerx/indexable_array.h"

#include <array>
#include <cstdint>
#include <numeric>

#include <gtest/gtest.h>
#include <gsl/gsl>

#include "chainerx/strides.h"

namespace chainerx {
namespace {

TEST(IndexableArrayTest, Rank0) {
    int value = 3;
    IndexableArray<int, 0> indexable_array(&value, {});
    EXPECT_EQ(0, indexable_array.ndim());
    EXPECT_EQ(&value, indexable_array.data());
    EXPECT_EQ(value, indexable_array[nullptr]);
    const int64_t i = 0;
    EXPECT_EQ(value, indexable_array[&i]);
}

TEST(IndexableArrayTest, Rank1) {
    std::array<int, 3> values = {2, 3, 4};
    const Strides strides = {sizeof(values[0])};
    IndexableArray<int, 1> indexable_array(&values[0], strides);
    EXPECT_EQ(1, indexable_array.ndim());
    EXPECT_EQ(&values[0], indexable_array.data());

    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_EQ(gsl::at(values, i), indexable_array[&i]);
    }
}

TEST(IndexableArrayTest, Rank3) {
    std::array<int, 2 * 3 * 4> values{};
    std::iota(values.begin(), values.end(), 0);
    const int64_t elemsize = sizeof(values[0]);
    const Strides strides = {3 * 4 * elemsize, 4 * elemsize, elemsize};
    IndexableArray<int, 3> indexable_array(&values[0], strides);
    EXPECT_EQ(3, indexable_array.ndim());
    EXPECT_EQ(&values[0], indexable_array.data());

    std::array<int64_t, 3> index{};
    int64_t lin = 0;
    for (index[0] = 0; index[0] < 2; ++index[0]) {
        for (index[1] = 0; index[1] < 3; ++index[1]) {
            for (index[2] = 0; index[2] < 4; ++index[2]) {
                EXPECT_EQ(gsl::at(values, lin++), indexable_array[&index[0]]);
            }
        }
    }
}

TEST(DynamicIndexableArrayTest, Rank0) {
    int value = 3;
    IndexableArray<int> indexable_array(&value, {});
    EXPECT_EQ(0, indexable_array.ndim());
    EXPECT_EQ(&value, indexable_array.data());
    EXPECT_EQ(value, indexable_array[nullptr]);
    const int64_t i = 0;
    EXPECT_EQ(value, indexable_array[&i]);
}

TEST(DynamicIndexableArrayTest, Rank1) {
    std::array<int, 3> values = {2, 3, 4};
    const Strides strides = {sizeof(values[0])};
    IndexableArray<int> indexable_array(&values[0], strides);
    EXPECT_EQ(1, indexable_array.ndim());
    EXPECT_EQ(&values[0], indexable_array.data());

    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_EQ(gsl::at(values, i), indexable_array[&i]);
    }
}

TEST(DynamicIndexableArrayTest, Rank3) {
    std::array<int, 2 * 3 * 4> values{};
    std::iota(values.begin(), values.end(), 0);
    const int64_t elemsize = sizeof(values[0]);
    const Strides strides = {3 * 4 * elemsize, 4 * elemsize, elemsize};
    IndexableArray<int> indexable_array(&values[0], strides);
    EXPECT_EQ(3, indexable_array.ndim());
    EXPECT_EQ(&values[0], indexable_array.data());

    std::array<int64_t, 3> index{};
    int64_t lin = 0;
    for (index[0] = 0; index[0] < 2; ++index[0]) {
        for (index[1] = 0; index[1] < 3; ++index[1]) {
            for (index[2] = 0; index[2] < 4; ++index[2]) {
                EXPECT_EQ(gsl::at(values, lin++), indexable_array[&index[0]]);
            }
        }
    }
}

}  // namespace
}  // namespace chainerx
