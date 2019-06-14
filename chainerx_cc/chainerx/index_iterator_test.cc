#include "chainerx/index_iterator.h"

#include <array>
#include <cstdint>

#include <gtest/gtest.h>

namespace chainerx {
namespace {

TEST(IndexIteratorTest, Rank0) {
    IndexIterator<0> it(nullptr, 1, 0, 1);
    EXPECT_EQ(0, it.ndim());
    EXPECT_EQ(0, it.raw_index());
    (void)it.index()[0];  // no throw
    EXPECT_TRUE(static_cast<bool>(it));
    ++it;
    EXPECT_EQ(0, it.ndim());
    EXPECT_EQ(1, it.raw_index());
    EXPECT_FALSE(static_cast<bool>(it));
    --it;
    EXPECT_EQ(0, it.ndim());
    EXPECT_EQ(0, it.raw_index());
    EXPECT_TRUE(static_cast<bool>(it));

    ++it;
    it.Restart();
    EXPECT_EQ(0, it.ndim());
    EXPECT_EQ(0, it.raw_index());
    EXPECT_TRUE(static_cast<bool>(it));
    it.Restart(1);
    EXPECT_EQ(0, it.ndim());
    EXPECT_EQ(1, it.raw_index());
    EXPECT_FALSE(static_cast<bool>(it));

    IndexIterator<0> a(nullptr, 1, 0, 1);
    it.CopyIndex(a);  // no throw
}

TEST(IndexIteratorTest, Rank1) {
    const std::array<int64_t, 1> shape = {3};
    IndexIterator<1> it(&shape[0], 3, 0, 1);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(1, it.ndim());
        EXPECT_EQ(i, it.raw_index());
        EXPECT_EQ(i, it.index()[0]);
        EXPECT_TRUE(static_cast<bool>(it));
        ++it;
    }
    EXPECT_FALSE(static_cast<bool>(it));
    for (int i = 2; i >= 0; --i) {
        --it;
        EXPECT_EQ(1, it.ndim());
        EXPECT_EQ(i, it.raw_index());
        EXPECT_EQ(i, it.index()[0]);
        EXPECT_TRUE(static_cast<bool>(it));
    }
    EXPECT_TRUE(static_cast<bool>(it));

    ++it;
    it.Restart();
    EXPECT_EQ(1, it.ndim());
    EXPECT_EQ(0, it.raw_index());
    EXPECT_EQ(0, it.index()[0]);
    EXPECT_TRUE(static_cast<bool>(it));
    it.Restart(1);
    EXPECT_EQ(1, it.ndim());
    EXPECT_EQ(1, it.raw_index());
    EXPECT_EQ(1, it.index()[0]);
    EXPECT_TRUE(static_cast<bool>(it));

    IndexIterator<1> a(&shape[0], 3, 1, 1);
    it.CopyIndex(a);
    EXPECT_EQ(1, it.ndim());
    EXPECT_EQ(1, it.index()[0]);
    EXPECT_TRUE(static_cast<bool>(it));
}

TEST(IndexIteratorTest, Rank3) {
    const std::array<int64_t, 3> shape = {2, 3, 4};
    IndexIterator<3> it(&shape[0], 24, 0, 1);
    for (int i = 0; i < 24; ++i) {
        EXPECT_EQ(3, it.ndim());
        EXPECT_EQ(i, it.raw_index());
        EXPECT_EQ((i / 12) % 2, it.index()[0]);
        EXPECT_EQ((i / 4) % 3, it.index()[1]);
        EXPECT_EQ(i % 4, it.index()[2]);
        EXPECT_TRUE(static_cast<bool>(it));
        ++it;
    }
    EXPECT_FALSE(static_cast<bool>(it));
    for (int i = 23; i >= 0; --i) {
        --it;
        EXPECT_EQ(3, it.ndim());
        EXPECT_EQ(i, it.raw_index());
        EXPECT_EQ((i / 12) % 2, it.index()[0]);
        EXPECT_EQ((i / 4) % 3, it.index()[1]);
        EXPECT_EQ(i % 4, it.index()[2]);
        EXPECT_TRUE(static_cast<bool>(it));
    }
    EXPECT_TRUE(static_cast<bool>(it));

    ++it;
    it.Restart();
    EXPECT_EQ(3, it.ndim());
    EXPECT_EQ(0, it.raw_index());
    EXPECT_EQ(0, it.index()[0]);
    EXPECT_EQ(0, it.index()[1]);
    EXPECT_EQ(0, it.index()[2]);
    EXPECT_TRUE(static_cast<bool>(it));
    it.Restart(1);
    EXPECT_EQ(3, it.ndim());
    EXPECT_EQ(1, it.raw_index());
    EXPECT_EQ(0, it.index()[0]);
    EXPECT_EQ(0, it.index()[1]);
    EXPECT_EQ(1, it.index()[2]);
    EXPECT_TRUE(static_cast<bool>(it));

    const std::array<int64_t, 2> a_shape = {2, 3};
    const std::array<int64_t, 1> b_shape = {4};
    IndexIterator<2> a(&a_shape[0], 6, 1, 1);
    IndexIterator<1> b(&b_shape[0], 4, 1, 1);
    it.CopyIndex(a);
    it.CopyIndex(b, a.ndim());
    EXPECT_EQ(3, it.ndim());
    EXPECT_EQ(0, it.index()[0]);
    EXPECT_EQ(1, it.index()[1]);
    EXPECT_EQ(1, it.index()[2]);
    EXPECT_TRUE(static_cast<bool>(it));
}

TEST(DynamicIndexIteratorTest, Rank0) {
    IndexIterator<> it(nullptr, 0, 1, 0, 1);
    EXPECT_EQ(0, it.ndim());
    EXPECT_EQ(0, it.raw_index());
    EXPECT_EQ(0, it.index()[0]);
    EXPECT_TRUE(static_cast<bool>(it));
    ++it;
    EXPECT_EQ(0, it.ndim());
    EXPECT_EQ(1, it.raw_index());
    EXPECT_EQ(0, it.index()[0]);
    EXPECT_FALSE(static_cast<bool>(it));
    --it;
    EXPECT_EQ(0, it.ndim());
    EXPECT_EQ(0, it.raw_index());
    EXPECT_EQ(0, it.index()[0]);
    EXPECT_TRUE(static_cast<bool>(it));

    ++it;
    it.Restart();
    EXPECT_EQ(0, it.ndim());
    EXPECT_EQ(0, it.raw_index());
    EXPECT_EQ(0, it.index()[0]);
    EXPECT_TRUE(static_cast<bool>(it));
    it.Restart(1);
    EXPECT_EQ(0, it.ndim());
    EXPECT_EQ(1, it.raw_index());
    EXPECT_EQ(0, it.index()[0]);
    EXPECT_FALSE(static_cast<bool>(it));

    IndexIterator<0> a(nullptr, 1, 0, 1);
    it.CopyIndex(a);  // no throw
}

TEST(DynamicIndexIteratorTest, Rank1) {
    const std::array<int64_t, 1> shape = {3};
    IndexIterator<> it(&shape[0], 1, 3, 0, 1);
    for (int i = 0; i < 3; ++i) {
        EXPECT_EQ(1, it.ndim());
        EXPECT_EQ(i, it.raw_index());
        EXPECT_EQ(i, it.index()[0]);
        EXPECT_TRUE(static_cast<bool>(it));
        ++it;
    }
    EXPECT_FALSE(static_cast<bool>(it));
    for (int i = 2; i >= 0; --i) {
        --it;
        EXPECT_EQ(1, it.ndim());
        EXPECT_EQ(i, it.raw_index());
        EXPECT_EQ(i, it.index()[0]);
        EXPECT_TRUE(static_cast<bool>(it));
    }
    EXPECT_TRUE(static_cast<bool>(it));

    ++it;
    it.Restart();
    EXPECT_EQ(1, it.ndim());
    EXPECT_EQ(0, it.raw_index());
    EXPECT_EQ(0, it.index()[0]);
    EXPECT_TRUE(static_cast<bool>(it));
    it.Restart(1);
    EXPECT_EQ(1, it.ndim());
    EXPECT_EQ(1, it.raw_index());
    EXPECT_EQ(1, it.index()[0]);
    EXPECT_TRUE(static_cast<bool>(it));

    IndexIterator<1> a(&shape[0], 3, 1, 1);
    it.CopyIndex(a);
    EXPECT_EQ(1, it.ndim());
    EXPECT_EQ(1, it.index()[0]);
    EXPECT_TRUE(static_cast<bool>(it));
}

TEST(DynamicIndexIteratorTest, Rank3) {
    const std::array<int64_t, 3> shape = {2, 3, 4};
    IndexIterator<> it(&shape[0], 3, 24, 0, 1);
    for (int i = 0; i < 24; ++i) {
        EXPECT_EQ(3, it.ndim());
        EXPECT_EQ(i, it.raw_index());
        EXPECT_EQ((i / 12) % 2, it.index()[0]);
        EXPECT_EQ((i / 4) % 3, it.index()[1]);
        EXPECT_EQ(i % 4, it.index()[2]);
        EXPECT_TRUE(static_cast<bool>(it));
        ++it;
    }
    EXPECT_FALSE(static_cast<bool>(it));
    for (int i = 23; i >= 0; --i) {
        --it;
        EXPECT_EQ(3, it.ndim());
        EXPECT_EQ(i, it.raw_index());
        EXPECT_EQ((i / 12) % 2, it.index()[0]);
        EXPECT_EQ((i / 4) % 3, it.index()[1]);
        EXPECT_EQ(i % 4, it.index()[2]);
        EXPECT_TRUE(static_cast<bool>(it));
    }
    EXPECT_TRUE(static_cast<bool>(it));

    ++it;
    it.Restart();
    EXPECT_EQ(3, it.ndim());
    EXPECT_EQ(0, it.raw_index());
    EXPECT_EQ(0, it.index()[0]);
    EXPECT_EQ(0, it.index()[1]);
    EXPECT_EQ(0, it.index()[2]);
    EXPECT_TRUE(static_cast<bool>(it));
    it.Restart(1);
    EXPECT_EQ(3, it.ndim());
    EXPECT_EQ(1, it.raw_index());
    EXPECT_EQ(0, it.index()[0]);
    EXPECT_EQ(0, it.index()[1]);
    EXPECT_EQ(1, it.index()[2]);
    EXPECT_TRUE(static_cast<bool>(it));

    const std::array<int64_t, 2> a_shape = {2, 3};
    const std::array<int64_t, 1> b_shape = {4};
    IndexIterator<2> a(&a_shape[0], 6, 1, 1);
    IndexIterator<1> b(&b_shape[0], 4, 1, 1);
    it.CopyIndex(a);
    it.CopyIndex(b, a.ndim());
    EXPECT_EQ(3, it.ndim());
    EXPECT_EQ(0, it.index()[0]);
    EXPECT_EQ(1, it.index()[1]);
    EXPECT_EQ(1, it.index()[2]);
    EXPECT_TRUE(static_cast<bool>(it));
}

}  // namespace
}  // namespace chainerx
