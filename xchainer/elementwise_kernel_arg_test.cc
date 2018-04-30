#include "xchainer/elementwise_kernel_arg.h"

#include <cstdint>
#include <tuple>

#include <gtest/gtest.h>

#include "xchainer/array.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/context_session.h"

namespace xchainer {
namespace {

TEST(MakeElementwiseKernelArgTest, CompressAllDims) {
    testing::ContextSession context_session;

    using T = float;
    Array a = testing::BuildArray({3, 2, 5, 4}).WithLinearData<T>();

    ElementwiseKernelArg<T> arg = MakeElementwiseKernelArg<T>(a);

    Indexer indexer = arg.indexer;
    EXPECT_EQ(1, indexer.ndim());
    EXPECT_EQ(a.shape()[0] * a.shape()[1] * a.shape()[2] * a.shape()[3], indexer.shape()[0]);

    IndexableArray<T> iarray = std::get<0>(arg.iarrays);
    EXPECT_EQ(1, iarray.ndim());
    EXPECT_EQ(static_cast<int64_t>(sizeof(T)), iarray.strides()[0]);
}

TEST(MakeElementwiseKernelArgTest, CompressSomeDims) {
    testing::ContextSession context_session;

    using T = float;
    Array a = testing::BuildArray({3, 2, 5, 4}).WithLinearData<T>().WithPadding({0, 2, 0, 0});

    ElementwiseKernelArg<T> arg = MakeElementwiseKernelArg<T>(a);

    Indexer indexer = arg.indexer;
    EXPECT_EQ(2, indexer.ndim());
    EXPECT_EQ(a.shape()[0] * a.shape()[1], indexer.shape()[0]);
    EXPECT_EQ(a.shape()[2] * a.shape()[3], indexer.shape()[1]);

    IndexableArray<T> iarray = std::get<0>(arg.iarrays);
    EXPECT_EQ(2, iarray.ndim());
    EXPECT_EQ(a.strides()[1], iarray.strides()[0]);
    EXPECT_EQ(a.strides()[3], iarray.strides()[1]);
}

TEST(MakeElementwiseKernelArgTest, CompressUnitLengthDim) {
    testing::ContextSession context_session;

    using T = float;
    Array a = testing::BuildArray({3, 2, 1, 4}).WithLinearData<T>().WithPadding(1);

    ElementwiseKernelArg<T> arg = MakeElementwiseKernelArg<T>(a);

    Indexer indexer = arg.indexer;
    EXPECT_EQ(3, indexer.ndim());
    EXPECT_EQ(a.shape()[0], indexer.shape()[0]);
    EXPECT_EQ(a.shape()[1], indexer.shape()[1]);
    EXPECT_EQ(a.shape()[3], indexer.shape()[2]);

    IndexableArray<T> iarray = std::get<0>(arg.iarrays);
    EXPECT_EQ(3, iarray.ndim());
    EXPECT_EQ(a.strides()[0], iarray.strides()[0]);
    EXPECT_EQ(a.strides()[1], iarray.strides()[1]);
    EXPECT_EQ(a.strides()[3], iarray.strides()[2]);
}

TEST(MakeElementwiseKernelArgTest, CompressMultipleArrayDims) {
    testing::ContextSession context_session;

    using T = float;
    Array a = testing::BuildArray({3, 2, 5, 4}).WithLinearData<T>().WithPadding({0, 2, 0, 0});
    Array b = testing::BuildArray({3, 2, 5, 4}).WithLinearData<T>().WithPadding({0, 0, 1, 0});

    ElementwiseKernelArg<T, T> arg = MakeElementwiseKernelArg<T, T>(a, b);

    Indexer indexer = arg.indexer;
    EXPECT_EQ(3, indexer.ndim());
    EXPECT_EQ(a.shape()[0] * a.shape()[1], indexer.shape()[0]);
    EXPECT_EQ(a.shape()[2], indexer.shape()[1]);
    EXPECT_EQ(a.shape()[3], indexer.shape()[2]);

    IndexableArray<T> a_iarray = std::get<0>(arg.iarrays);
    EXPECT_EQ(3, a_iarray.ndim());
    EXPECT_EQ(a.strides()[1], a_iarray.strides()[0]);
    EXPECT_EQ(a.strides()[2], a_iarray.strides()[1]);
    EXPECT_EQ(a.strides()[3], a_iarray.strides()[2]);

    IndexableArray<T> b_iarray = std::get<1>(arg.iarrays);
    EXPECT_EQ(3, b_iarray.ndim());
    EXPECT_EQ(b.strides()[1], b_iarray.strides()[0]);
    EXPECT_EQ(b.strides()[2], b_iarray.strides()[1]);
    EXPECT_EQ(b.strides()[3], b_iarray.strides()[2]);
}

}  // namespace
}  // namespace xchainer
