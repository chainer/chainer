#include "xchainer/indexer.h"

#include <gtest/gtest.h>

namespace xchainer {
namespace {

TEST(IndexerTest, Ctor) {
    Indexer<0> indexer({});
    EXPECT_EQ(0, indexer.ndim());
    EXPECT_EQ(1, indexer.total_size());
    EXPECT_NO_THROW(indexer.Set(0));
}

TEST(IndexerTest, Rank1) {
    Indexer<1> indexer({3});
    EXPECT_EQ(1, indexer.ndim());
    EXPECT_EQ(3, indexer.total_size());
    EXPECT_EQ(3, indexer.shape()[0]);

    for (int64_t i = 0; i < 3; ++i) {
        indexer.Set(i);
        EXPECT_EQ(i, indexer.index()[0]);
        EXPECT_EQ(i, indexer.raw_index());
    }
}

TEST(IndexerTest, Rank3) {
    Indexer<3> indexer({2, 3, 4});
    EXPECT_EQ(3, indexer.ndim());
    EXPECT_EQ(2 * 3 * 4, indexer.total_size());

    EXPECT_EQ(2, indexer.shape()[0]);
    EXPECT_EQ(3, indexer.shape()[1]);
    EXPECT_EQ(4, indexer.shape()[2]);

    int64_t raw_i = 0;
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            for (int64_t k = 0; k < 4; ++k) {
                indexer.Set(raw_i);
                EXPECT_EQ(i, indexer.index()[0]);
                EXPECT_EQ(j, indexer.index()[1]);
                EXPECT_EQ(k, indexer.index()[2]);
                EXPECT_EQ(raw_i, indexer.raw_index());
                raw_i++;
            }
        }
    }
}

TEST(DynamicIndexerTest, Rank0) {
    Indexer<> indexer({});
    EXPECT_EQ(0, indexer.ndim());
    EXPECT_EQ(1, indexer.total_size());
    EXPECT_NO_THROW(indexer.Set(0));
}

TEST(DynamicIndexerTest, Rank1) {
    Indexer<> indexer({3});
    EXPECT_EQ(1, indexer.ndim());
    EXPECT_EQ(3, indexer.total_size());
    EXPECT_EQ(3, indexer.shape()[0]);

    for (int64_t i = 0; i < 3; ++i) {
        indexer.Set(i);
        EXPECT_EQ(i, indexer.index()[0]);
        EXPECT_EQ(i, indexer.raw_index());
    }
}

TEST(DynamicIndexerTest, Rank3) {
    Indexer<> indexer({2, 3, 4});
    EXPECT_EQ(3, indexer.ndim());
    EXPECT_EQ(2 * 3 * 4, indexer.total_size());

    EXPECT_EQ(2, indexer.shape()[0]);
    EXPECT_EQ(3, indexer.shape()[1]);
    EXPECT_EQ(4, indexer.shape()[2]);

    int64_t raw_i = 0;
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t j = 0; j < 3; ++j) {
            for (int64_t k = 0; k < 4; ++k) {
                indexer.Set(raw_i);
                EXPECT_EQ(i, indexer.index()[0]);
                EXPECT_EQ(j, indexer.index()[1]);
                EXPECT_EQ(k, indexer.index()[2]);
                EXPECT_EQ(raw_i, indexer.raw_index());
                raw_i++;
            }
        }
    }
}

}  // namespace
}  // namespace xchainer
